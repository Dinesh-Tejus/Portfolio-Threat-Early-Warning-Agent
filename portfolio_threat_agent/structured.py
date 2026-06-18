from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Materiality(str, Enum):
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"


class Source(BaseModel):
    source_id: str
    title: str
    url: str
    query: str
    content: str = ""
    raw_content: str | None = None
    published_date: date | None = None
    score: float | None = None

    @property
    def evidence_text(self) -> str:
        return self.content or self.raw_content or self.title


class Claim(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)


class MarketQuote(BaseModel):
    ticker: str
    price: float
    currency: str = "USD"


class PriceReaction(BaseModel):
    ticker: str
    catalyst_date: date
    active_date: date
    start_price: float
    end_price: float
    pct_change: float
    currency: str = "USD"


class ThreatSignal(BaseModel):
    ticker: str
    event_type: str
    headline: str
    summary: str
    claims: list[Claim]
    source_ids: list[str]
    catalyst_date: date | None = None
    materiality: Materiality = Materiality.LOW
    materiality_reason: str = ""
    position_value: float | None = None
    portfolio_weight: float | None = None
    distance_to_stop_pct: float | None = None
    price_reaction_pct: float | None = None
    price_reaction_start: float | None = None
    price_reaction_end: float | None = None


class CitationIssue(BaseModel):
    signal_index: int
    claim_index: int | None = None
    issue: str
    source_id: str | None = None
    reason: str | None = None


class StructuredBrief(BaseModel):
    active_date: date
    sources: list[Source]
    signals: list[ThreatSignal]
    blocked_issues: list[CitationIssue] = Field(default_factory=list)
