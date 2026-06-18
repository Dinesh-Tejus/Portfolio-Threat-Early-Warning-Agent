from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from langgraph.errors import GraphRecursionError
from langsmith import traceable
from pydantic import BaseModel, Field

from portfolio_threat_agent.agent import (
    combine_validated_results,
    configure_langsmith,
    create_portfolio_threat_agent,
    run_portfolio_entries,
)
from portfolio_threat_agent.models import DEFAULT_MODEL
from portfolio_threat_agent.portfolio import load_portfolio
from portfolio_threat_agent.structured import Materiality


class ExpectedEvent(BaseModel):
    ticker: str
    event_type: str
    catalyst_date: date
    expected_materiality: Materiality
    reference_sources: list[str] = Field(default_factory=list)
    notes: str | None = None


class EvalCase(BaseModel):
    id: str
    portfolio_path: str
    start_date: date
    end_date: date
    expected_events: list[ExpectedEvent] = Field(default_factory=list)
    instruction: str | None = None


class MatchedEvent(BaseModel):
    ticker: str
    event_type: str
    catalyst_date: date
    materiality: Materiality


class MaterialityMismatch(BaseModel):
    ticker: str
    event_type: str
    catalyst_date: date
    expected_materiality: Materiality
    actual_materiality: Materiality


class UnexpectedSignal(BaseModel):
    ticker: str
    event_type: str
    catalyst_date: date | None = None
    materiality: Materiality
    headline: str = ""
    source_urls: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    id: str
    passed: bool
    matched_events: list[MatchedEvent] = Field(default_factory=list)
    missing_events: list[ExpectedEvent] = Field(default_factory=list)
    materiality_mismatches: list[MaterialityMismatch] = Field(default_factory=list)
    unexpected_high_medium_signals: list[UnexpectedSignal] = Field(default_factory=list)
    citation_issues: list[dict[str, Any]] = Field(default_factory=list)
    rendered_brief: str
    agent_error: str | None = None


class EvalReport(BaseModel):
    total: int
    passed: int
    results: list[EvalResult]


def load_eval_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("cases", [])
    return [EvalCase.model_validate(item) for item in payload]


_EVENT_TYPE_SYNONYMS: dict[str, set[str]] = {
    "guidance": {"guidance", "guidance cut", "earnings guidance", "outlook cut", "outlook", "revenue guidance", "sales guidance", "product guidance", "forecast cut", "guidance revision"},
    "downgrade": {"downgrade", "analyst downgrade", "rating downgrade", "price target cut", "rating cut", "analyst rating"},
    "regulatory": {"regulatory", "regulatory action", "faa", "sec", "fda", "export controls", "sanctions", "grounding", "recall", "regulatory shock", "import ban"},
    "lawsuit": {"lawsuit", "doj lawsuit", "antitrust", "legal action", "litigation", "court", "sue", "sued", "legal"},
    "earnings": {"earnings", "earnings miss", "earnings miss/guidance", "quarterly results", "q1", "q2", "q3", "q4"},
    "acquisition": {"acquisition", "merger", "m&a", "takeover", "buyout"},
    "macro": {"macro", "sector", "sector pressure", "sector/macro pressure", "macro pressure", "market sell-off"},
}


def _normalize_event_type(event_type: str) -> str:
    normalized = event_type.strip().lower()
    for canonical, synonyms in _EVENT_TYPE_SYNONYMS.items():
        if normalized == canonical or normalized in synonyms:
            return canonical
    return normalized


def _signal_date(signal: dict[str, Any]) -> date | None:
    raw = signal.get("catalyst_date")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _find_matching_signal(
    expected: "ExpectedEvent",
    signals: list[dict[str, Any]],
    date_tolerance_days: int = 1,
) -> dict[str, Any] | None:
    expected_ticker = expected.ticker.strip().upper()
    expected_type = _normalize_event_type(expected.event_type)
    for signal in signals:
        if str(signal.get("ticker", "")).strip().upper() != expected_ticker:
            continue
        if _normalize_event_type(str(signal.get("event_type", ""))) != expected_type:
            continue
        actual_date = _signal_date(signal)
        if actual_date is None:
            continue
        if abs((expected.catalyst_date - actual_date).days) <= date_tolerance_days:
            return signal
    return None


def _is_expected_signal(signal: dict[str, Any], expected_events: "list[ExpectedEvent]", date_tolerance_days: int = 1) -> bool:
    ticker = str(signal.get("ticker", "")).strip().upper()
    event_type = _normalize_event_type(str(signal.get("event_type", "")))
    actual_date = _signal_date(signal)
    for event in expected_events:
        if event.ticker.strip().upper() != ticker:
            continue
        if _normalize_event_type(event.event_type) != event_type:
            continue
        if actual_date is not None and abs((event.catalyst_date - actual_date).days) <= date_tolerance_days:
            return True
    return False


def _materiality(value: Any) -> Materiality:
    return Materiality(str(value).strip().upper())


def _unexpected_signal(signal: dict[str, Any]) -> UnexpectedSignal:
    catalyst_date = None
    if signal.get("catalyst_date"):
        catalyst_date = date.fromisoformat(str(signal["catalyst_date"])[:10])
    return UnexpectedSignal(
        ticker=str(signal.get("ticker", "")).upper(),
        event_type=str(signal.get("event_type", "")).lower(),
        catalyst_date=catalyst_date,
        materiality=_materiality(signal.get("materiality")),
        headline=str(signal.get("headline") or ""),
        source_urls=[str(url) for url in signal.get("source_urls", []) if url],
    )


@traceable(run_type="chain", name="run_eval_case")
def run_eval_case(case: EvalCase, model: str = DEFAULT_MODEL, max_results: int = 5) -> EvalResult:
    configure_langsmith()
    portfolio = load_portfolio(Path(case.portfolio_path))
    agent = create_portfolio_threat_agent(
        model=model,
        max_results=max_results,
        start_date=case.start_date,
        end_date=case.end_date,
    )
    try:
        _messages, results = run_portfolio_entries(
            agent,
            portfolio,
            since="7d",
            instruction=case.instruction,
            active_date=case.end_date,
            retrieval_start=case.start_date,
        )
    except GraphRecursionError as exc:
        return EvalResult(
            id=case.id,
            passed=False,
            missing_events=case.expected_events,
            rendered_brief="No answer produced: agent exceeded the graph step limit before finishing this eval case.",
            agent_error=f"graph_recursion_limit: {exc}",
        )
    if not results:
        return EvalResult(
            id=case.id,
            passed=False,
            missing_events=case.expected_events,
            rendered_brief="No answer produced: agent did not produce validated candidate results.",
            agent_error="missing_validated_results",
        )

    rendered_brief = combine_validated_results(results)
    signals: list[dict[str, Any]] = []
    citation_issues: list[dict[str, Any]] = []
    statuses: list[str] = []
    for result in results:
        statuses.append(str(result.get("status") or ""))
        structured = result.get("structured_brief")
        if isinstance(structured, dict):
            signals.extend(signal for signal in structured.get("signals", []) if isinstance(signal, dict))
        citation_issues.extend(issue for issue in result.get("blocked_issues", []) if isinstance(issue, dict))

    matched_events: list[MatchedEvent] = []
    missing_events: list[ExpectedEvent] = []
    materiality_mismatches: list[MaterialityMismatch] = []

    for event in case.expected_events:
        signal = _find_matching_signal(event, signals)
        if signal is None:
            missing_events.append(event)
            continue

        actual_materiality = _materiality(signal.get("materiality"))
        if actual_materiality == event.expected_materiality:
            matched_events.append(
                MatchedEvent(
                    ticker=event.ticker,
                    event_type=event.event_type,
                    catalyst_date=event.catalyst_date,
                    materiality=actual_materiality,
                )
            )
        else:
            materiality_mismatches.append(
                MaterialityMismatch(
                    ticker=event.ticker,
                    event_type=event.event_type,
                    catalyst_date=event.catalyst_date,
                    expected_materiality=event.expected_materiality,
                    actual_materiality=actual_materiality,
                )
            )

    unexpected_high_medium = []
    for signal in signals:
        if not isinstance(signal, dict) or _is_expected_signal(signal, case.expected_events):
            continue
        sig = _unexpected_signal(signal)
        if sig.materiality in {Materiality.HIGH, Materiality.MED}:
            unexpected_high_medium.append(sig)
    passed = not missing_events and not materiality_mismatches and not citation_issues
    return EvalResult(
        id=case.id,
        passed=passed,
        matched_events=matched_events,
        missing_events=missing_events,
        materiality_mismatches=materiality_mismatches,
        unexpected_high_medium_signals=unexpected_high_medium,
        citation_issues=citation_issues,
        rendered_brief=rendered_brief,
        agent_error=None if all(status == "ok" for status in statuses) else ",".join(statuses),
    )


@traceable(run_type="chain", name="run_eval_dataset")
def run_eval_dataset(path: Path, model: str = DEFAULT_MODEL, max_results: int = 5) -> EvalReport:
    results = [run_eval_case(case, model=model, max_results=max_results) for case in load_eval_cases(path)]
    return EvalReport(total=len(results), passed=sum(1 for result in results if result.passed), results=results)
