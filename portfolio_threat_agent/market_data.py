from __future__ import annotations

from datetime import date, timedelta

from langsmith import traceable
import yfinance as yf
from portfolio_threat_agent.structured import MarketQuote, PriceReaction


def _fast_info_value(fast_info: object, key: str) -> object | None:
    if hasattr(fast_info, "get"):
        return fast_info.get(key)  # type: ignore[attr-defined]
    return getattr(fast_info, key, None)


def _history_closes(history: object) -> list[tuple[date, float]]:
    close = history["Close"].dropna()  # type: ignore[index]
    rows: list[tuple[date, float]] = []
    for index, value in close.items():
        close_date = index.date() if hasattr(index, "date") else date.fromisoformat(str(index)[:10])
        rows.append((close_date, float(value)))
    return rows


@traceable(run_type="tool", name="get_yfinance_quote")
def get_yfinance_quote(ticker: str) -> MarketQuote:
    """Fetch the latest Yahoo Finance quote with a recent-close fallback."""


    symbol = ticker.strip().upper()
    t = yf.Ticker(symbol)
    fi = t.fast_info
    price = _fast_info_value(fi, "last_price")
    currency = _fast_info_value(fi, "currency") or "USD"

    if price is None:
        history = t.history(period="5d", interval="1d")
        if history.empty:
            raise ValueError(f"No recent Yahoo Finance price found for {symbol}")
        price = _history_closes(history)[-1][1]

    return MarketQuote(ticker=symbol, price=float(price), currency=str(currency))


@traceable(run_type="tool", name="get_yfinance_quote_as_of")
def get_yfinance_quote_as_of(ticker: str, active_date: date) -> MarketQuote:
    """Fetch the nearest prior daily close at or before active_date."""
    if active_date == date.today():
        return get_yfinance_quote(ticker)

    symbol = ticker.strip().upper()
    t = yf.Ticker(symbol)
    fi = t.fast_info
    currency = _fast_info_value(fi, "currency") or "USD"
    history = t.history(
        start=(active_date - timedelta(days=10)).isoformat(),
        end=(active_date + timedelta(days=1)).isoformat(),
        interval="1d",
    )
    if history.empty:
        raise ValueError(f"No Yahoo Finance history found for {symbol} at or before {active_date}")

    closes = _history_closes(history)
    candidates = [(day, price) for day, price in closes if day <= active_date]
    if not candidates:
        raise ValueError(f"No Yahoo Finance close found for {symbol} at or before {active_date}")
    return MarketQuote(ticker=symbol, price=float(candidates[-1][1]), currency=str(currency))


@traceable(run_type="tool", name="get_yfinance_price_reaction")
def get_yfinance_price_reaction(ticker: str, catalyst_date: date, active_date: date) -> PriceReaction:
    """Fetch price change from the prior trading close before a catalyst to active/latest price."""
    symbol = ticker.strip().upper()
    if active_date < catalyst_date:
        raise ValueError(f"active_date {active_date} is before catalyst_date {catalyst_date}")

    t = yf.Ticker(symbol)
    fi = t.fast_info
    currency = _fast_info_value(fi, "currency") or "USD"
    history = t.history(
        start=(catalyst_date - timedelta(days=10)).isoformat(),
        end=(active_date + timedelta(days=2)).isoformat(),
        interval="1d",
    )
    if history.empty:
        raise ValueError(f"No Yahoo Finance history found for {symbol} around {catalyst_date}")

    closes = _history_closes(history)
    start_candidates = [(day, price) for day, price in closes if day < catalyst_date]
    if not start_candidates:
        start_candidates = [(day, price) for day, price in closes if day <= catalyst_date]
    if not start_candidates:
        start_candidates = closes[:1]
    start_price = start_candidates[-1][1]

    end_candidates = [(day, price) for day, price in closes if day <= active_date]
    if not end_candidates:
        end_candidates = closes[-1:]
    end_price = end_candidates[-1][1]
    if active_date == date.today():
        live_price = _fast_info_value(fi, "last_price")
        if live_price is not None:
            end_price = float(live_price)

    if start_price == 0:
        raise ValueError(f"Cannot compute price reaction for {symbol}: start price is zero")
    pct_change = ((end_price - start_price) / start_price) * 100
    return PriceReaction(
        ticker=symbol,
        catalyst_date=catalyst_date,
        active_date=active_date,
        start_price=float(start_price),
        end_price=float(end_price),
        pct_change=float(pct_change),
        currency=str(currency),
    )
