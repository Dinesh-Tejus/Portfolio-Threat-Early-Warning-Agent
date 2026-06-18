# PRD — Portfolio Threat Early-Warning Agent

**One-liner:** A portfolio-aware agent that scans the open web for *risk-relevant* catalysts on your holdings and returns ranked, source-cited threat alerts — stop-aware, no predictions, no advice.

**Status:** Final Draft · **Context:** Optimized for Agent Engineering & Evaluation

---

## 1. Problem

A holder of N positions can't manually monitor every name for events that threaten their book. General finance news ranks by *newsworthiness*; brokerage "news for your holdings" tabs are an unranked firehose of every headline mentioning a ticker. Neither knows **how material an event is to *this* portfolio**, and neither ties findings to **how close a position sits to its stop**.

## 2. Goal / Non-Goals

**Goals**
- Detect discrete, explainable catalysts that could threaten a position (downgrade, lawsuit, guidance cut, regulatory action, M&A, sector shock).
- Rank by materiality to *this* book (position size, sector concentration, recency, distance-to-stop).
- Ground every claim in retrieved source text. Grounding is enforced by a deterministic citation contract (§5, §8), along with prompting.

**Non-Goals**
- Predict price movement.
- Issue buy/sell recommendations or stop adjustments.
- Serve as a full portfolio monitor (no P&L, no price/quote analytics beyond context).
- Technical/trend signals (this is catalyst-driven, not chart-driven).

## 3. Scope Boundary

| Agent **does** | Agent **does not** |
|---|---|
| Surface a catalyst | Predict its price impact |
| Report attributed source sentiment ("sources read this negative [1]") | State its own directional verdict |
| Compute distance-to-stop as **context** | Forecast that a stop will trigger |
| Rank by materiality + reasoning | Recommend an action |

*News is good at discrete threat events. It is **not** a source of price/quote data — those come from a market-data API, never web scraping.*

---

## 4. Input & Evaluation Harness

**Portfolio file** (`portfolio.json`):
```json
{
  "positions": [
    { "ticker": "NVDA", "shares": 120, "stop_price": 158.00 },
    { "ticker": "PFE",  "shares": 400, "stop_price": 24.50 }
  ]
}
```

**Run Commands:**
```bash
# Live monitoring
portfolio-threat-agent monitor --portfolio portfolio.json --since 7d

# One-off question
portfolio-threat-agent ask --portfolio portfolio.json "Check NVDA for risk catalysts"

# Historical evaluation (bounds Tavily and Yahoo Finance to the date window)
portfolio-threat-agent eval --dataset examples/eval_cases.json
```

- `--since`: Bounds retrieval window (default `7d`). Supports `7d`, `4w`, `3m`, `1y`.
- `--start-date` / `--end-date`: Historical date bounds for retrieval and market data. Used by the eval harness to prevent truth-leakage.
- `--virtual-date`: Compatibility alias for `--end-date`.

---

## 5. Output

**User-facing:** A ranked threat brief per position.
```
HIGH · NVDA (8.2% of book · 4.1% above stop)
  2 analysts cut price targets this week on margin concerns. [1][2]
  Exposure: $41,000 · Distance to stop: 4.1% above stop · Why HIGH: top-3 position by quoted position value + within 10% of stop + <48h old
  Market reaction since catalyst: -3.2% ($184.00 -> $178.10)
  [1] Reuters, 2024-01-02 https://...
  [2] Bloomberg, 2024-01-02 https://...
```

**Structured Object (The Eval Contract):**
Every `claim.text` must map to ≥1 valid `source_id`. A claim with no valid source is a caught error, not silent text. Citation issues are recorded in `blocked_issues` and the signal is excluded from the final brief.

---

## 5b. Materiality (Definition & Scoring)

**Materiality** = how much a given event threatens *this* portfolio. It is a per-signal score, **not** a market-wide judgment — the same event can be HIGH for one book and absent for another. It is the agent's core wedge and its only ranking axis.

**Inputs**

| Input | Meaning | Direction |
|---|---|---|
| Exposure | Position size as % of quoted book value | larger → higher |
| Proximity to stop | Distance between current price and `stop_price` | closer → higher |
| Concentration | Event hits an overweight sector (>25% of quoted book) | overweight → higher |
| Recency | Age of the catalyst | newer → higher |

**Scoring method: rule tiers** (transparent, no weights — chosen over weighted scoring so the logic is defensible and `materiality_reason` falls out directly).

```
HIGH  if (top-3 position by quoted book value)
      AND (within 10% of stop OR no stop set)
      AND (catalyst < 48h old)

MED   if any TWO of:
        - top-5 position by quoted book value
        - within 20% of stop
        - catalyst < 7d old
        - event hits an overweight sector (>25% of book)

LOW   otherwise (still a real, sourced threat — just lower priority)
```

Every signal carries a `materiality_reason` listing the rules that fired.

Position ranks and sector concentration are computed from live Yahoo Finance quotes when available, falling back to share count and position count respectively when quotes are unavailable.

> **Note:** This rule block is intentionally isolated as a single scoring function so the logic can be swapped e.g. to weighted scoring or a learned ranker without touching retrieval, extraction, or rendering.

---

## 6. Architecture

```
portfolio.json
     │
     ▼
[1] Portfolio Loader ──► positions + stops
     │
     ▼
[2] LangChain Agent ──► decides query wording, Tavily parameters, and search angles
     │                   (analyst action, company event, price-move explanation, sector/macro)
     │
     ├──► [3] tavily_search tool ──► auto_parameters=True, include_answer=False
     │                               application enforces start_date/end_date bounds
     │                               search budget: max 4 calls per holding run
     │
     └──► [4] validate_candidate_threat tool ──► LLM judge checks:
               │                                  (1) source supports the claim
               │                                  (2) claim is explicit negative news
               │                                  (3) event is specific and actionable
               ▼
[5] Validated Candidate Assembly ──► approved tool outputs only
     │                                parses Tavily messages into Source objects
     │                                dedupes by URL
     ▼
[6] Yahoo Finance Enrichment (app-side) ──► quotes, book weight, stop distance, price reaction
     │
     ▼
[7] Materiality Ranker ──► score = f(exposure, concentration, recency, distance_to_stop)
     │                       attach materiality_reason
     ▼
[8] Citation Contract ──► every claim source_id must exist in retrieved sources
     │                     signals with unknown source_ids are blocked
     ▼
[9] Structured Brief (Pydantic) ──► StructuredBrief with signals, sources, blocked_issues
     │
     ▼
[10] Renderer ──► CLI / Telegram
```

The key architectural boundary is between the **agent loop** ([2]–[4]) and the **validated brief builder** ([5]–[9]). The agent loop is flexible and exploratory. The brief builder is contract-driven and conservative.

---

## 7. Evaluation Loop

| Metric | Type | What it proves |
|---|---|---|
| **Historical Recall** | Backtest | Did the agent find the actual catalyst for a known past event within the retrieval window? |
| **Citation Coverage** | Deterministic | Every claim maps to a real retrieved source with a known `source_id`. |
| **Faithfulness / Threatness** | LLM-as-judge | Each claim is (1) supported by its cited source excerpt and (2) concrete negative news, not a passing mention or neutral event. |
| **Materiality Accuracy** | Golden set | Does the agent assign the expected HIGH/MED/LOW to annotated events? |

**Eval dataset:** 5 historical cases in `examples/eval_cases.json`, using `examples/golden_portfolio.json` (6 positions: AMD, MRNA, TSLA, AAPL, NKE, BA). Each case specifies `start_date`, `end_date`, expected ticker/event/catalyst_date/materiality, and an optional search instruction.

**Pass criteria per case:**
- All expected events matched by ticker, event type, and catalyst date (±1 day tolerance).
- No materiality mismatches on expected events.
- No citation issues on expected signals.
- Unexpected HIGH/MED signals from other portfolio tickers do not fail the case — the full portfolio is always scanned.

**Run:**
```bash
portfolio-threat-agent eval --dataset examples/eval_cases.json
```

---

## 8. Truth-Leakage Defense (Backtest Integrity)

In evaluation mode the model has pretraining knowledge of how past events resolved. Prompting it to "respect the virtual date" is a **soft control on a hard problem** and is treated as a hint, not a guarantee. Integrity is enforced **mechanically**, in layers:

1. **Retrieval cap (hard).** Tavily `end_date` is anchored to the case `end_date`, so no future-dated document can enter context.
2. **Grounding contract (hard, primary defense).** The eval scores claims against **retrieved sources, not against the model's answer**. Every claim must cite a source from the current run; any claim sourced from model memory has no valid citation and **fails the contract automatically**.
3. **Yahoo Finance as-of prices (hard).** Market enrichment uses the nearest prior daily close at or before `end_date`, not the live price.
4. **Prompt time-boxing (soft, last).** System prompt instructs reliance only on retrieved context. Used as reinforcement, never as the sole control.

**Stance:** leakage is controlled by the grounding contract, retrieval cap, and as-of market data.

---

## 9. Integration & Risks

- **LLM:** `gpt-4o-mini` via `langchain-openai`. Requires `OPENAI_API_KEY`.
- **Retrieval:** Tavily via `langchain-tavily`. `auto_parameters=True`, `include_answer=False`, quality domain allow-list, quote-page block-list. Requires `TAVILY_API_KEY`.
- **Price Data:** Yahoo Finance (`yfinance`), app-side only after agent validation. Never fetched as an agent tool. Required for `distance_to_stop` and `portfolio_weight`; if absent, signals are still surfaced with those fields marked unavailable.
- **Tracing:** LangSmith (`langsmith`). Optional. Set `LANGSMITH_API_KEY` to enable. Project: `portfolio-threat-agent`.
- **Persistence:** Monitor config and alert state stored in local JSON files under `state/` by default. Set `MONGODB_URI` to use MongoDB instead.
- **Delivery:** CLI (streaming) and Telegram bot (long-polling). Requires `TELEGRAM_BOT_TOKEN` for Telegram.
- **Risk (Truth Leak):** Handled mechanically per §8.

---

## 10. Success Criteria

- Correct retrieval and materiality on ≥3 of 5 historical eval cases.
- Citation coverage: no approved signal with an unresolvable `source_id`.
- Backtest integrity demonstrably enforced by the grounding contract and retrieval cap.
- Traceable runs via LangSmith showing agent searches, validation decisions, and materiality ranking.
