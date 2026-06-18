from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

from langsmith import traceable

from portfolio_threat_agent.portfolio import Portfolio
from portfolio_threat_agent.structured import CitationIssue, MarketQuote, Materiality, PriceReaction, Source, StructuredBrief, ThreatSignal


def price_reaction_key(ticker: str, catalyst_date: date) -> str:
    return f"{ticker.strip().upper()}|{catalyst_date.isoformat()}"


def _position_values(portfolio: Portfolio, quotes: dict[str, MarketQuote]) -> dict[str, float]:
    return {
        position.ticker: position.shares * quotes[position.ticker].price
        for position in portfolio.positions
        if position.ticker in quotes
    }


def _position_ranks(portfolio: Portfolio, position_values: dict[str, float]) -> tuple[dict[str, int], str]:
    if position_values:
        def _rank_key(position: Any) -> tuple:
            val = position_values.get(position.ticker)
            return (val is not None, val or 0, position.shares)

        ordered = sorted(portfolio.positions, key=_rank_key, reverse=True)
        return {position.ticker: index + 1 for index, position in enumerate(ordered)}, "quoted position value"

    ordered = sorted(portfolio.positions, key=lambda position: position.shares, reverse=True)
    return {position.ticker: index + 1 for index, position in enumerate(ordered)}, "share count fallback"


def _sector_concentration(portfolio: Portfolio, position_values: dict[str, float]) -> tuple[dict[str, float], str]:
    totals: dict[str, float] = defaultdict(float)
    if position_values:
        for position in portfolio.positions:
            value = position_values.get(position.ticker)
            if value is not None:
                totals[position.sector or "unknown"] += value
        total_value = sum(totals.values()) or 1
        return {sector: value / total_value for sector, value in totals.items()}, "quoted book value"

    for position in portfolio.positions:
        totals[position.sector or "unknown"] += 1
    total_count = sum(totals.values()) or 1
    return {sector: count / total_count for sector, count in totals.items()}, "position count fallback"


@traceable(run_type="chain", name="rank_materiality")
def rank_materiality(
    portfolio: Portfolio,
    signals: list[ThreatSignal],
    active_date: date,
    quotes: dict[str, MarketQuote] | None = None,
    price_reactions: dict[str, PriceReaction] | None = None,
) -> list[ThreatSignal]:
    quotes = quotes or {}
    price_reactions = price_reactions or {}
    positions = {position.ticker: position for position in portfolio.positions}
    sectors = {position.ticker: position.sector or "unknown" for position in portfolio.positions}
    position_values = _position_values(portfolio, quotes)
    concentrations, concentration_basis = _sector_concentration(portfolio, position_values)
    ranks, rank_basis = _position_ranks(portfolio, position_values)
    total_value = sum(position_values.values())
    ranked: list[ThreatSignal] = []

    for signal in signals:
        position = positions.get(signal.ticker)
        rank = ranks.get(signal.ticker, 999)
        sector = sectors.get(signal.ticker, "unknown")
        sector_overweight = concentrations.get(sector, 0) > 0.25
        position_value = position_values.get(signal.ticker)
        portfolio_weight = position_value / total_value if position_value is not None and total_value else None
        quote = quotes.get(signal.ticker)
        distance_to_stop_pct = None
        if quote and position and position.stop_price:
            distance_to_stop_pct = ((quote.price - position.stop_price) / position.stop_price) * 100
        below_stop = distance_to_stop_pct is not None and distance_to_stop_pct < 0
        near_10 = distance_to_stop_pct is not None and distance_to_stop_pct <= 10
        near_20 = distance_to_stop_pct is not None and distance_to_stop_pct <= 20
        no_stop = position is not None and position.stop_price is None
        days_old = (active_date - signal.catalyst_date).days if signal.catalyst_date else None
        recent_48h = days_old is not None and 0 <= days_old < 2
        recent_7d = days_old is None or 0 <= days_old < 7

        reasons: list[str] = []
        if rank <= 3:
            reasons.append(f"top-3 position by {rank_basis}")
        elif rank <= 5:
            reasons.append(f"top-5 position by {rank_basis}")
        if sector_overweight:
            reasons.append(f"overweight sector/theme by {concentration_basis}")
        if portfolio_weight is not None:
            reasons.append(f"{portfolio_weight:.1%} of quoted book")
        if below_stop:
            reasons.append("below stop")
        elif near_10:
            reasons.append("within 10% of stop")
        elif near_20:
            reasons.append("within 20% of stop")
        elif no_stop:
            reasons.append("no stop set")
        if recent_48h:
            reasons.append("<48h old")
        elif recent_7d:
            reasons.append("<7d old or undated catalyst")
        if distance_to_stop_pct is None:
            reasons.append("stop distance unavailable")
        reaction = price_reactions.get(price_reaction_key(signal.ticker, signal.catalyst_date)) if signal.catalyst_date else None
        if reaction:
            if reaction.pct_change <= -5:
                reasons.append("price down >=5% since catalyst")
            elif reaction.pct_change < 0:
                reasons.append("price down since catalyst")

        med_count = sum([rank <= 5, near_20, recent_7d, sector_overweight])
        if rank <= 3 and (near_10 or no_stop) and recent_48h:
            materiality = Materiality.HIGH
        elif med_count >= 2:
            materiality = Materiality.MED
        else:
            materiality = Materiality.LOW

        ranked.append(
            signal.model_copy(
                update={
                    "materiality": materiality,
                    "materiality_reason": " + ".join(reasons),
                    "position_value": position_value,
                    "portfolio_weight": portfolio_weight,
                    "distance_to_stop_pct": distance_to_stop_pct,
                    "price_reaction_pct": reaction.pct_change if reaction else None,
                    "price_reaction_start": reaction.start_price if reaction else None,
                    "price_reaction_end": reaction.end_price if reaction else None,
                }
            )
        )

    order = {Materiality.HIGH: 0, Materiality.MED: 1, Materiality.LOW: 2}
    return sorted(ranked, key=lambda signal: (order[signal.materiality], ranks.get(signal.ticker, 999), signal.catalyst_date or date.min))


@traceable(run_type="chain", name="enforce_citation_contract")
def enforce_citation_contract(
    signals: list[ThreatSignal],
    sources: list[Source],
) -> tuple[list[ThreatSignal], list[CitationIssue]]:
    source_by_id = {source.source_id: source for source in sources}
    valid: list[ThreatSignal] = []
    issues: list[CitationIssue] = []

    def validate_source(signal_index: int, source_id: str, claim_index: int | None = None) -> CitationIssue | None:
        source = source_by_id.get(source_id)
        if source is None:
            issue = "unknown claim source_id" if claim_index is not None else "unknown signal source_id"
            return CitationIssue(signal_index=signal_index, claim_index=claim_index, issue=issue, source_id=source_id)
        return None

    for signal_index, signal in enumerate(signals):
        signal_issues: list[CitationIssue] = []
        for source_id in signal.source_ids:
            issue = validate_source(signal_index, source_id)
            if issue is not None:
                signal_issues.append(issue)

        for claim_index, claim in enumerate(signal.claims):
            if not claim.source_ids:
                signal_issues.append(CitationIssue(signal_index=signal_index, claim_index=claim_index, issue="claim has no source_ids"))
            for source_id in claim.source_ids:
                issue = validate_source(signal_index, source_id, claim_index=claim_index)
                if issue is not None:
                    signal_issues.append(issue)

        if signal_issues:
            issues.extend(signal_issues)
        else:
            valid.append(signal)
    return valid, issues


@traceable(run_type="chain", name="build_structured_brief")
def build_structured_brief(
    portfolio: Portfolio,
    sources: list[Source],
    signals: list[ThreatSignal],
    active_date: date,
    quotes: dict[str, MarketQuote] | None = None,
    price_reactions: dict[str, PriceReaction] | None = None,
) -> StructuredBrief:
    ranked = rank_materiality(portfolio, signals, active_date, quotes=quotes, price_reactions=price_reactions)
    valid, issues = enforce_citation_contract(ranked, sources)
    return StructuredBrief(active_date=active_date, sources=sources, signals=valid, blocked_issues=issues)


def _stop_context(distance_to_stop_pct: float | None) -> str:
    if distance_to_stop_pct is None:
        return "stop distance unavailable"
    if distance_to_stop_pct < 0:
        return f"{abs(distance_to_stop_pct):.1f}% below stop"
    return f"{distance_to_stop_pct:.1f}% above stop"


def _source_label(source: Source) -> str:
    date_text = f", {source.published_date}" if source.published_date else ""
    return f"{source.title}{date_text} {source.url}"


def _price_reaction_context(signal: ThreatSignal) -> str | None:
    if signal.price_reaction_pct is None or signal.price_reaction_start is None or signal.price_reaction_end is None:
        return None
    return (
        f"Market reaction since catalyst: {signal.price_reaction_pct:+.1f}% "
        f"(${signal.price_reaction_start:,.2f} -> ${signal.price_reaction_end:,.2f})"
    )


def render_structured_brief(brief: StructuredBrief) -> str:
    if not brief.signals:
        return "No sourced, citation-valid threat alerts found."

    source_by_id = {source.source_id: source for source in brief.sources}
    lines: list[str] = []
    for signal in brief.signals:
        weight = f"{signal.portfolio_weight:.1%} of book" if signal.portfolio_weight is not None else "unknown book weight"
        lines.append(f"{signal.materiality.value} · {signal.ticker} ({weight} · {_stop_context(signal.distance_to_stop_pct)})")

        source_numbers: dict[str, int] = {}
        ordered_source_ids: list[str] = []
        for claim in signal.claims:
            for source_id in claim.source_ids:
                if source_id in source_by_id and source_id not in source_numbers:
                    source_numbers[source_id] = len(source_numbers) + 1
                    ordered_source_ids.append(source_id)
            markers = "".join(f"[{source_numbers[source_id]}]" for source_id in claim.source_ids if source_id in source_numbers)
            lines.append(f"  {claim.text} {markers}".rstrip())

        exposure = f"Exposure: ${signal.position_value:,.0f}" if signal.position_value is not None else "Exposure: unavailable"
        lines.append(f"  {exposure} · Distance to stop: {_stop_context(signal.distance_to_stop_pct)} · Why {signal.materiality.value}: {signal.materiality_reason}")
        reaction_context = _price_reaction_context(signal)
        if reaction_context:
            lines.append(f"  {reaction_context}")

        for source_id in ordered_source_ids:
            source = source_by_id[source_id]
            lines.append(f"  [{source_numbers[source_id]}] {_source_label(source)}")
        lines.append("")
    return "\n".join(lines).strip()
