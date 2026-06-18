from __future__ import annotations

import json
import os
from pathlib import Path
import re
from datetime import date, timedelta
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langsmith import traceable

from portfolio_threat_agent.models import DEFAULT_MODEL, create_chat_model
from portfolio_threat_agent.portfolio import Portfolio, format_portfolio_for_agent, load_portfolio
from portfolio_threat_agent.source_parsing import _tool_name, extract_tavily_sources
from portfolio_threat_agent.telegram import send_telegram_message
from portfolio_threat_agent.tools import (
    build_brief_from_validated_candidates,
    build_tavily_search_tool,
    reset_tavily_search_budget,
    start_tavily_search_budget,
    validate_candidate_threat,
)
from portfolio_threat_agent.prompts import ASK_INSTRUCTION_TEMPLATE, PORTFOLIO_INSTRUCTION_TEMPLATE, SYSTEM_PROMPT

load_dotenv()


StreamHandler = Callable[[str], None]
AGENT_RECURSION_LIMIT = 24



@traceable(run_type="prompt", name="build_portfolio_instruction_for_portfolio")
def build_portfolio_instruction_for_portfolio(
    portfolio: Portfolio,
    since: str = "7d",
    extra_instruction: str | None = None,
    active_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> str:
    active = end_date or active_date or date.today()
    start = start_date or window_start_date(since, active)
    extra_block = f"\nAdditional user instruction:\n{extra_instruction}\n" if extra_instruction else ""
    return PORTFOLIO_INSTRUCTION_TEMPLATE.format(
        portfolio=format_portfolio_for_agent(portfolio),
        portfolio_json=portfolio.model_dump_json(),
        active_date=active.isoformat(),
        start_date=start.isoformat() if start else "unbounded",
        end_date=active.isoformat(),
        extra_instruction_block=extra_block,
    ).strip()


@traceable(run_type="prompt", name="build_portfolio_instruction")
def build_portfolio_instruction(
    portfolio_path: Path,
    since: str = "7d",
    extra_instruction: str | None = None,
    active_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> str:
    return build_portfolio_instruction_for_portfolio(
        load_portfolio(portfolio_path),
        since=since,
        extra_instruction=extra_instruction,
        active_date=active_date,
        start_date=start_date,
        end_date=end_date,
    )


def window_start_date(since: str, active_date: date) -> date | None:
    match = re.fullmatch(r"\s*(\d+)\s*([dDwWmMyY])\s*", since)
    if not match:
        return None
    count = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "d":
        return active_date - timedelta(days=count)
    if unit == "w":
        return active_date - timedelta(days=count * 7)
    if unit == "m":
        return active_date - timedelta(days=count * 30)
    return active_date - timedelta(days=count * 365)


def configure_langsmith() -> None:
    os.environ.setdefault("LANGSMITH_PROJECT", "portfolio-threat-agent")
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    else:
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


@traceable(run_type="chain", name="create_portfolio_threat_agent")
def create_portfolio_threat_agent(
    model: str = DEFAULT_MODEL,
    max_results: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
):
    chat_model = create_chat_model(model=model, temperature=0)
    search_tool = build_tavily_search_tool(max_results=max_results, start_date=start_date, end_date=end_date)
    return create_agent(
        model=chat_model,
        tools=[search_tool, validate_candidate_threat],
        system_prompt=SYSTEM_PROMPT,
    )


@traceable(run_type="parser", name="message_text")
def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _parse_json_content(message: Any) -> Any:
    text = message_text(message)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _shorten(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _stream_message_events(chunk: Any) -> list[tuple[str | None, Any]]:
    events: list[tuple[str | None, Any]] = []
    if not isinstance(chunk, dict):
        return events
    if isinstance(chunk.get("messages"), list):
        events.extend((None, message) for message in chunk["messages"])
    for node, value in chunk.items():
        if isinstance(value, dict) and isinstance(value.get("messages"), list):
            events.extend((str(node), message) for message in value["messages"])
    return events


def _format_tool_args(args: Any) -> str:
    if not isinstance(args, dict):
        return _shorten(str(args), 160)
    if "query" in args:
        rendered = [f"query={str(args.get('query'))!r}"]
        for key in (
            "topic",
            "search_depth",
            "time_range",
            "days",
            "start_date",
            "end_date",
            "include_domains",
            "exclude_domains",
            "max_results",
        ):
            value = args.get(key)
            if value not in (None, "", [], {}):
                rendered.append(f"{key}={value!r}")
        return _shorten(", ".join(rendered), 220)
    if {"source_url", "claim"} & set(args):
        rendered = []
        for key in ("ticker", "event_type", "source_url", "active_date"):
            value = args.get(key)
            if value not in (None, ""):
                rendered.append(f"{key}={value!r}")
        if args.get("claim"):
            rendered.append(f"claim={_shorten(str(args.get('claim')), 100)!r}")
        return _shorten(", ".join(rendered), 220)
    if "ticker" in args:
        rendered = [f"ticker={str(args.get('ticker')).upper()!r}"]
        for key in ("catalyst_date", "active_date"):
            value = args.get(key)
            if value not in (None, ""):
                rendered.append(f"{key}={value!r}")
        return ", ".join(rendered)
    if "text" in args:
        return f"text={_shorten(str(args.get('text')), 140)!r}"
    if not args:
        return ""
    return _shorten(json.dumps(args, default=str), 160)


def _tool_call_text(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return _shorten(str(tool_call), 180)
    name = tool_call.get("name") or "unknown_tool"
    args = tool_call.get("args") or {}
    rendered_args = _format_tool_args(args)
    return f"{name}({rendered_args})" if rendered_args else f"{name}()"


@traceable(run_type="parser", name="format_agent_stream_event")
def format_agent_stream_event(message: Any) -> str | None:
    message_type = getattr(message, "type", "")
    if message_type == "ai":
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            return "agent called " + "; ".join(_tool_call_text(call) for call in tool_calls)
        text = message_text(message).strip()
        if not text:
            return None
        return f"agent draft: {_shorten(text, 220)}"

    if message_type == "tool":
        name = _tool_name(message) or "tool"
        payload = _parse_json_content(message)
        if name == "tavily_search" and isinstance(payload, dict):
            query = payload.get("query") or "search"
            if payload.get("error"):
                return f"tavily_search skipped for {query!r}: {_shorten(str(payload.get('error')), 160)}"
            results = payload.get("results")
            result_count = len(results) if isinstance(results, list) else 0
            result_label = "result" if result_count == 1 else "results"
            return f"tavily_search returned {result_count} {result_label} for {query!r}"
        if name == "validate_candidate_threat" and isinstance(payload, dict):
            ticker = payload.get("ticker")
            status = payload.get("status")
            reason = payload.get("reason") or payload.get("feedback") or ""
            return f"validate_candidate_threat {status} for {ticker}: {_shorten(str(reason), 160)}"
        return f"{name} returned {_shorten(message_text(message), 220)}"

    return None


@traceable(run_type="chain", name="ask_portfolio_threat_agent")
def run_agent_messages(agent: Any, question: str) -> list[Any]:
    return run_agent_messages_stream(agent, question)


@traceable(run_type="chain", name="stream_portfolio_threat_agent")
def run_agent_messages_stream(agent: Any, question: str, stream_handler: StreamHandler | None = None) -> list[Any]:
    messages: list[Any] = []
    budget_token = start_tavily_search_budget()
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
            stream_mode="updates",
        ):
            for _, message in _stream_message_events(chunk):
                messages.append(message)
                event = format_agent_stream_event(message)
                if event and stream_handler:
                    stream_handler(event)
    finally:
        reset_tavily_search_budget(budget_token)
    return messages


@traceable(run_type="parser", name="extract_agent_explanation")
def extract_agent_explanation(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = message_text(message).strip()
            if text:
                return text
    return ""


def validated_signal_count(results: list[dict[str, object]]) -> int:
    return sum(int(result.get("signal_count") or 0) for result in results)


def validated_source_urls(results: list[dict[str, object]]) -> set[str]:
    urls: set[str] = set()
    for result in results:
        source_urls = result.get("source_urls")
        if isinstance(source_urls, list):
            urls.update(str(url) for url in source_urls if url)
    return urls


def combine_validated_results(results: list[dict[str, object]]) -> str:
    rendered_with_signals = [
        str(result.get("rendered_brief") or "").strip()
        for result in results
        if int(result.get("signal_count") or 0) > 0 and str(result.get("rendered_brief") or "").strip()
    ]
    if rendered_with_signals:
        return "\n\n".join(rendered_with_signals)
    if results:
        return "No sourced, citation-valid threat alerts found."
    return "No answer produced: agent did not produce validated candidate results."


@traceable(run_type="chain", name="run_portfolio_entries")
def run_portfolio_entries(
    agent: Any,
    portfolio: Portfolio,
    since: str,
    instruction: str | None,
    active_date: date,
    retrieval_start: date | None,
    stream_handler: StreamHandler | None = None,
) -> tuple[list[Any], list[dict[str, object]]]:
    all_messages: list[Any] = []
    results: list[dict[str, object]] = []
    for position in portfolio.positions:
        single_position_portfolio = Portfolio(positions=[position])
        if stream_handler:
            stream_handler(f"checking {position.ticker}")
        prompt = build_portfolio_instruction_for_portfolio(
            single_position_portfolio,
            since=since,
            extra_instruction=instruction,
            active_date=active_date,
            start_date=retrieval_start,
            end_date=active_date,
        )
        messages = run_agent_messages_stream(agent, prompt, stream_handler=stream_handler) if stream_handler else run_agent_messages(agent, prompt)
        all_messages.extend(messages)
        results.append(build_brief_from_validated_candidates(portfolio, active_date, messages))
    return all_messages, results


@traceable(run_type="chain", name="ask_portfolio_threat_agent")
def ask_agent(
    question: str,
    portfolio_path: Path,
    since: str = "7d",
    model: str = DEFAULT_MODEL,
    max_results: int = 5,
    virtual_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    stream_handler: StreamHandler | None = None,
    notify: bool = True,
) -> str:
    configure_langsmith()
    active_date = end_date or virtual_date or date.today()
    retrieval_start = start_date or window_start_date(since, active_date)
    portfolio = load_portfolio(portfolio_path)
    agent = create_portfolio_threat_agent(
        model=model,
        max_results=max_results,
        start_date=retrieval_start,
        end_date=active_date,
    )
    prompt = ASK_INSTRUCTION_TEMPLATE.format(
        portfolio=format_portfolio_for_agent(portfolio),
        portfolio_json=portfolio.model_dump_json(),
        active_date=active_date.isoformat(),
        start_date=retrieval_start.isoformat() if retrieval_start else "unbounded",
        end_date=active_date.isoformat(),
        question=question,
    )
    messages = run_agent_messages_stream(agent, prompt, stream_handler=stream_handler)
    sources = extract_tavily_sources(messages)
    if not sources:
        return "No answer produced: agent did not search Tavily."
    return extract_agent_explanation(messages) or "No answer produced."


@traceable(run_type="chain", name="monitor_portfolio")
def monitor_portfolio(
    portfolio_path: Path,
    since: str = "7d",
    instruction: str | None = None,
    model: str = DEFAULT_MODEL,
    max_results: int = 5,
    virtual_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    stream_handler: StreamHandler | None = None,
    notify: bool = True,
) -> str:
    configure_langsmith()
    active_date = end_date or virtual_date or date.today()
    retrieval_start = start_date or window_start_date(since, active_date)
    portfolio = load_portfolio(portfolio_path)
    agent = create_portfolio_threat_agent(
        model=model,
        max_results=max_results,
        start_date=retrieval_start,
        end_date=active_date,
    )
    messages, results = run_portfolio_entries(
        agent,
        portfolio,
        since=since,
        instruction=instruction,
        active_date=active_date,
        retrieval_start=retrieval_start,
        stream_handler=stream_handler,
    )
    sources = extract_tavily_sources(messages)
    if not sources:
        return "No answer produced: agent did not search Tavily."
    if not results:
        explanation = extract_agent_explanation(messages)
        base = "No answer produced: agent did not produce validated candidate results."
        return f"{base}\n\nAgent said:\n{explanation}" if explanation else base
    rendered = combine_validated_results(results)
    if notify and active_date == date.today() and validated_signal_count(results) > 0:
        send_telegram_message(rendered)
    return rendered
