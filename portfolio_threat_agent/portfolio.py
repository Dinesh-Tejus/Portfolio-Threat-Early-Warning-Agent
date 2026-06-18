from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from langsmith import traceable


class Position(BaseModel):
    ticker: str
    shares: float = Field(gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    sector: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class Portfolio(BaseModel):
    positions: list[Position]

    @model_validator(mode="after")
    def require_positions(self) -> "Portfolio":
        if not self.positions:
            raise ValueError("portfolio must include at least one position")
        return self


@traceable(run_type="chain", name="load_portfolio")
def load_portfolio(path: Path) -> Portfolio:
    return Portfolio.model_validate(json.loads(path.read_text()))


@traceable(run_type="parser", name="format_portfolio_for_agent")
def format_portfolio_for_agent(portfolio: Portfolio) -> str:
    lines = []
    for position in portfolio.positions:
        sector = position.sector or "unknown sector"
        stop = position.stop_price if position.stop_price is not None else "not set"
        lines.append(f"- {position.ticker}: {position.shares:g} shares, stop={stop}, sector={sector}")
    return "\n".join(lines)
