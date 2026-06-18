from __future__ import annotations

import json
import re
import threading
from contextvars import ContextVar, Token
from datetime import date
from collections.abc import Callable
from typing import Any, Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langchain_tavily.tavily_search import TavilySearchInput
from langsmith import traceable

from portfolio_threat_agent.market_data import get_yfinance_price_reaction, get_yfinance_quote, get_yfinance_quote_as_of
from portfolio_threat_agent.models import DEFAULT_MODEL, create_chat_model
from portfolio_threat_agent.portfolio import Portfolio
from portfolio_threat_agent.prompts import VALIDATE_CANDIDATE_SYSTEM_PROMPT
from portfolio_threat_agent.source_parsing import _message_text, _tool_name, extract_tavily_sources
from portfolio_threat_agent.structured import Claim, MarketQuote, PriceReaction, ThreatSignal
from portfolio_threat_agent.threats import build_structured_brief, price_reaction_key, render_structured_brief
from pydantic import BaseModel, Field

CandidateValidationVerdict = Literal["approved", "unsupported", "not_threat"]


class CandidateValidationDecision(BaseModel):
    verdict: CandidateValidationVerdict
    reason: str = Field(default="")


CandidateValidationJudge = Callable[[str, str, str, str], CandidateValidationDecision]


INCLUDE_DOMAINS = [
    "reuters.com", "apnews.com", "wsj.com", "ft.com",
    "marketwatch.com", "investopedia.com", "bloomberg.com",
    "axios.com", "seekingalpha.com", "thestreet.com",
    "cnbc.com", "finance.yahoo.com", "businessinsider.com",
    "theguardian.com", "nytimes.com", "forbes.com",
]

EXCLUDE_DOMAINS = [
    "ca.finance.yahoo.com/quote",
    "finance.yahoo.com/quote",
]

DEFAULT_SEARCH_BUDGET = 4


class SearchBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0
        self.lock = threading.Lock()

    def consume(self) -> bool:
        with self.lock:
            if self.count >= self.limit:
                return False
            self.count += 1
            return True


_SEARCH_BUDGET: ContextVar[SearchBudget | None] = ContextVar("portfolio_threat_search_budget", default=None)


def start_tavily_search_budget(limit: int = DEFAULT_SEARCH_BUDGET) -> Token[SearchBudget | None]:
    return _SEARCH_BUDGET.set(SearchBudget(limit))


def reset_tavily_search_budget(token: Token[SearchBudget | None]) -> None:
    _SEARCH_BUDGET.reset(token)


def _consume_tavily_search_budget() -> bool:
    budget = _SEARCH_BUDGET.get()
    return True if budget is None else budget.consume()


@traceable(run_type="tool", name="build_tavily_search_tool")
def build_tavily_search_tool(max_results: int = 5, start_date: date | None = None, end_date: date | None = None):
    """Build the web-search tool the agent can call."""
    tavily = TavilySearch(
        max_results=max_results,
        auto_parameters=True,
        include_raw_content=False,
        include_answer=False,
        include_domains=INCLUDE_DOMAINS,
        exclude_domains=EXCLUDE_DOMAINS,
    )

    start_text = start_date.isoformat() if start_date else None
    end_text = end_date.isoformat() if end_date else None

    def bounded_tavily_search(**kwargs: Any) -> Any:
        call_args = dict(kwargs)
        query = str(call_args.get("query") or "")
        if not _consume_tavily_search_budget():
            return {
                "query": query,
                "results": [],
                "error": f"search_budget_exhausted: max {DEFAULT_SEARCH_BUDGET} Tavily searches per holding run",
            }
        if start_text:
            requested_start = call_args.get("start_date")
            if not requested_start or str(requested_start) < start_text:
                call_args["start_date"] = start_text
        if end_text:
            requested_end = call_args.get("end_date")
            if not requested_end or str(requested_end) > end_text:
                call_args["end_date"] = end_text
        return tavily.invoke(call_args)

    return StructuredTool.from_function(
        name="tavily_search",
        description=(
            f"{tavily.description}\n\n"
            f"Retrieval window is enforced by the application: start_date >= {start_text or 'unbounded'}, "
            f"end_date <= {end_text or 'unbounded'}. "
            f"Search budget is enforced by the application: max {DEFAULT_SEARCH_BUDGET} Tavily searches per holding run."
        ),
        func=bounded_tavily_search,
        args_schema=TavilySearchInput,
    )


def _runtime_messages(runtime: ToolRuntime | None) -> list[Any]:
    if runtime is None:
        return []
    state = getattr(runtime, "state", None)
    if isinstance(state, dict) and isinstance(state.get("messages"), list):
        return state["messages"]
    return []


def _parse_json_text(text: str) -> Any:
    if not isinstance(text, str):
        return text if isinstance(text, (dict, list)) else None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _source_excerpt(source: Any, limit: int = 4000) -> str:
    text = " ".join(str(source.evidence_text).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _source_for_url(messages: list[Any], source_url: str):
    for source in extract_tavily_sources(messages):
        if source.url == source_url:
            return source
    return None


def _prior_validation_for(messages: list[Any], ticker: str, source_url: str, event_type: str) -> dict[str, object] | None:
    normalized_ticker = ticker.strip().upper()
    normalized_event = event_type.strip().lower()
    for message in messages:
        if getattr(message, "type", None) != "tool" or "validate_candidate_threat" not in _tool_name(message):
            continue
        payload = _parse_json_text(_message_text(message))
        if not isinstance(payload, dict):
            continue
        if str(payload.get("ticker", "")).strip().upper() != normalized_ticker:
            continue
        if str(payload.get("source_url", "")).strip() != source_url.strip():
            continue
        if str(payload.get("event_type", "")).strip().lower() != normalized_event:
            continue
        return payload
    return None


def _structured_brief_payload(brief: Any) -> dict[str, object]:
    source_by_id = {source.source_id: source for source in brief.sources}
    signals: list[dict[str, object]] = []
    for signal in brief.signals:
        payload = signal.model_dump(mode="json")
        payload["source_urls"] = [source_by_id[source_id].url for source_id in signal.source_ids if source_id in source_by_id]
        signals.append(payload)
    return {
        "active_date": brief.active_date.isoformat(),
        "signals": signals,
        "sources": [source.model_dump(mode="json") for source in brief.sources],
        "blocked_issues": [issue.model_dump(mode="json") for issue in brief.blocked_issues],
    }


def _parse_candidate_validation_decision(text: str) -> CandidateValidationDecision:
    payload = _parse_json_text(text)
    if not isinstance(payload, dict):
        return CandidateValidationDecision(verdict="unsupported", reason="Judge did not return JSON.")
    verdict = str(payload.get("verdict", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if verdict not in {"approved", "unsupported", "not_threat"}:
        return CandidateValidationDecision(verdict="unsupported", reason=f"Judge returned invalid verdict: {verdict or 'missing'}")
    return CandidateValidationDecision(verdict=verdict, reason=reason)


def create_candidate_validation_judge(model: str = DEFAULT_MODEL) -> CandidateValidationJudge:
    chat_model = create_chat_model(model=model, temperature=0)

    @traceable(run_type="llm", name="judge_candidate_validation")
    def judge(ticker: str, event_type: str, claim: str, source_excerpt: str) -> CandidateValidationDecision:
        response = chat_model.invoke(
            [
                ("system", VALIDATE_CANDIDATE_SYSTEM_PROMPT),
                (
                    "human",
                    f"Ticker: {ticker}\n"
                    f"Candidate event type: {event_type}\n"
                    f"Claim: {claim}\n\n"
                    "Source excerpt:\n"
                    f"{source_excerpt}\n\n"
                    'Return JSON only, e.g. {"verdict":"approved","reason":"..."}',
                ),
            ]
        )
        return _parse_candidate_validation_decision(_message_text(response))

    return judge


@traceable(run_type="tool", name="validate_candidate_threat_from_messages")
def validate_candidate_threat_from_messages(
    ticker: str,
    event_type: str,
    claim: str,
    source_url: str,
    active_date: str,
    messages: list[Any],
    catalyst_date: str | None = None,
    model: str = DEFAULT_MODEL,
    candidate_judge: CandidateValidationJudge | None = None,
) -> dict[str, object]:
    """Validate that a candidate is source-supported negative threat news."""
    symbol = ticker.strip().upper()
    source = _source_for_url(messages, source_url)
    if source is None:
        return {
            "status": "rejected",
            "reason": "source_url was not found in Tavily results for this agent run",
            "ticker": symbol,
            "event_type": event_type,
            "claim": claim,
            "source_url": source_url,
        }

    prior_validation = _prior_validation_for(messages, symbol, source_url, event_type)
    if prior_validation is not None:
        return {
            "status": "duplicate",
            "reason": "this ticker/source/event was already validated in this agent run",
            "ticker": symbol,
            "event_type": event_type.strip().lower() or "other",
            "claim": claim,
            "source_url": source_url,
            "previous_status": prior_validation.get("status"),
            "previous_reason": prior_validation.get("reason") or prior_validation.get("feedback", ""),
        }

    excerpt = _source_excerpt(source)
    decision = (candidate_judge or create_candidate_validation_judge(model))(symbol, event_type, claim, excerpt)
    if decision.verdict == "unsupported":
        return {
            "status": "rejected",
            "reason": "claim is unsupported by cited source",
            "feedback": decision.reason,
            "ticker": symbol,
            "event_type": event_type,
            "claim": claim,
            "source_url": source_url,
            "source": source.model_dump(mode="json"),
        }

    if decision.verdict == "not_threat":
        return {
            "status": "rejected",
            "reason": "candidate is not concrete negative threat news",
            "feedback": decision.reason,
            "ticker": symbol,
            "event_type": event_type,
            "claim": claim,
            "source_url": source_url,
            "source": source.model_dump(mode="json"),
        }

    try:
        fallback_catalyst_date = date.fromisoformat(active_date[:10])
    except ValueError:
        fallback_catalyst_date = date.today()
    parsed_catalyst_date = source.published_date or fallback_catalyst_date
    if catalyst_date:
        try:
            parsed_catalyst_date = date.fromisoformat(catalyst_date[:10])
        except ValueError:
            parsed_catalyst_date = source.published_date or fallback_catalyst_date

    return {
        "status": "approved",
        "ticker": symbol,
        "event_type": event_type.strip().lower() or "other",
        "claim": claim,
        "headline": source.title,
        "source_url": source.url,
        "source_id": source.source_id,
        "catalyst_date": parsed_catalyst_date.isoformat(),
        "source": source.model_dump(mode="json"),
        "reason": decision.reason,
    }


@tool
def validate_candidate_threat(
    ticker: str,
    event_type: str,
    claim: str,
    source_url: str,
    active_date: str,
    runtime: ToolRuntime,
    catalyst_date: str | None = None,
) -> str:
    """Validate that a Tavily-sourced candidate is a supported, concrete negative threat catalyst for the holding.

    IMPORTANT — claim must be a direct quote or close paraphrase of text explicitly present in the Tavily result for
    source_url. Do not add facts, figures, or specifics from training knowledge. source_url must be a URL that
    appeared in a Tavily search result during this run; never use a URL from memory."""
    result = validate_candidate_threat_from_messages(
        ticker=ticker,
        event_type=event_type,
        claim=claim,
        source_url=source_url,
        active_date=active_date,
        catalyst_date=catalyst_date,
        messages=_runtime_messages(runtime),
        model=DEFAULT_MODEL,
    )
    return json.dumps(result)


def _validated_candidates_from_messages(messages: list[Any]) -> tuple[list[ThreatSignal], list[Any]]:
    signals: list[ThreatSignal] = []
    sources_by_id: dict[str, Any] = {}
    seen: set[tuple[str, str, str]] = set()
    source_map = {s.url: s for s in extract_tavily_sources(messages)}
    for message in messages:
        if getattr(message, "type", None) != "tool" or "validate_candidate_threat" not in _tool_name(message):
            continue
        payload = _parse_json_text(_message_text(message))
        if not isinstance(payload, dict) or payload.get("status") != "approved":
            continue
        source = source_map.get(payload.get("source_url", ""))
        if source is None:
            continue
        ticker = str(payload["ticker"]).strip().upper()
        event_type = str(payload["event_type"]).strip().lower() or "other"
        claim = str(payload["claim"]).strip()
        key = (ticker, event_type, source.url)
        if not claim or key in seen:
            continue
        seen.add(key)
        sources_by_id[source.source_id] = source
        catalyst = source.published_date
        if payload.get("catalyst_date"):
            try:
                catalyst = date.fromisoformat(str(payload["catalyst_date"])[:10])
            except ValueError:
                catalyst = source.published_date
        signals.append(
            ThreatSignal(
                ticker=ticker,
                event_type=event_type,
                headline=str(payload.get("headline") or source.title),
                summary=claim,
                claims=[Claim(text=claim, source_ids=[source.source_id])],
                source_ids=[source.source_id],
                catalyst_date=catalyst,
            )
        )
    return signals, list(sources_by_id.values())


@traceable(run_type="chain", name="build_brief_from_validated_candidates")
def build_brief_from_validated_candidates(
    portfolio: Portfolio,
    active_date: date,
    messages: list[Any],
) -> dict[str, object]:
    """Build the final brief from agent-validated candidates, then enrich with Yahoo Finance data."""
    signals, sources = _validated_candidates_from_messages(messages)
    if not signals:
        return {
            "status": "ok",
            "rendered_brief": "No sourced, validated threat alerts found.",
            "structured_brief": {"active_date": active_date.isoformat(), "signals": [], "sources": [], "blocked_issues": []},
            "signal_count": 0,
            "source_urls": [],
            "blocked_issues": [],
        }

    quotes: dict[str, MarketQuote] = {}
    price_reactions: dict[str, PriceReaction] = {}
    signal_tickers = {signal.ticker for signal in signals}
    portfolio_tickers = {position.ticker for position in portfolio.positions}
    for ticker in sorted(portfolio_tickers | signal_tickers):
        try:
            quote = get_yfinance_quote_as_of(ticker, active_date) if active_date != date.today() else get_yfinance_quote(ticker)
            quotes[ticker] = quote
        except Exception:
            continue

    for signal in sorted(signals, key=lambda s: (s.ticker, str(s.catalyst_date))):
        if not signal.catalyst_date:
            continue
        try:
            reaction = get_yfinance_price_reaction(signal.ticker, catalyst_date=signal.catalyst_date, active_date=active_date)
            price_reactions[price_reaction_key(signal.ticker, signal.catalyst_date)] = reaction
        except Exception:
            continue

    brief = build_structured_brief(
        portfolio,
        sources,
        signals,
        active_date=active_date,
        quotes=quotes,
        price_reactions=price_reactions,
    )
    rendered = render_structured_brief(brief)
    used_source_ids = {source_id for signal in brief.signals for source_id in signal.source_ids}
    source_urls = [source.url for source in brief.sources if source.source_id in used_source_ids]
    return {
        "status": "ok",
        "rendered_brief": rendered,
        "structured_brief": _structured_brief_payload(brief),
        "signal_count": len(brief.signals),
        "source_urls": source_urls,
        "blocked_issues": [issue.model_dump(mode="json") for issue in brief.blocked_issues],
    }
