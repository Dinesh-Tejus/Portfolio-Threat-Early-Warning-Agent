# Setup Guide

This guide is only about getting the project running: installation, credentials, commands, Telegram, MongoDB, evaluation, and tests. For internals, see [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md). For the design story, see [APPROACH_AND_BUILD_STORY.md](APPROACH_AND_BUILD_STORY.md).

## 1. Requirements

- Python `>=3.11`
- Tavily API key
- OpenAI API key
- Optional LangSmith API key for tracing
- Optional Telegram bot token for chat commands and alert delivery
- Optional MongoDB URI for monitor config/state persistence

The project runs locally as a Python package.

## 2. Install

From the project root:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Commands can be run as either:

```bash
portfolio-threat-agent ...
python -m portfolio_threat_agent ...
```

## 3. Environment

Create `.env` in the project root or export values in your shell:

```bash
OPENAI_API_KEY="..."
TAVILY_API_KEY="tvly-..."
```

Optional LangSmith tracing:

```bash
LANGSMITH_API_KEY="lsv2_..."
LANGSMITH_PROJECT="portfolio-threat-agent"
```

Optional Telegram:

```bash
TELEGRAM_BOT_TOKEN="..."
TELEGRAM_CHAT_ID="..."   # optional fallback; usually learned from Telegram messages
```

Notes:

- The current code default is `gpt-4o-mini`.
- `OPENAI_API_KEY` is required. There is no fallback provider.
- `TAVILY_API_KEY` is required for live retrieval.
- Without `LANGSMITH_API_KEY`, tracing env vars default off.

## 4. Portfolio JSON

Example:

```json
{
  "positions": [
    { "ticker": "NVDA", "shares": 120, "stop_price": 158.0, "sector": "Semiconductors" },
    { "ticker": "PFE", "shares": 400, "stop_price": 24.5, "sector": "Pharmaceuticals" }
  ]
}
```

Required:

- `ticker`
- `shares`

Optional:

- `stop_price`
- `sector`

`stop_price` is used for risk context and materiality ranking. The system does not recommend stop changes.

## 5. One-Time Portfolio Check

```bash
python -m portfolio_threat_agent monitor \
  --portfolio examples/portfolio.json \
  --since 7d
```

What happens:

- each holding is processed separately
- the agent chooses Tavily searches for that holding
- Tavily generated answers are disabled
- raw page Markdown is not requested for live runs
- candidate threats are validated against retrieved Tavily snippets
- approved candidates are enriched with Yahoo Finance market data
- the final structured brief is printed locally

The CLI streams concise progress events before the final brief.

## 6. Portfolio-Scoped Ask

```bash
python -m portfolio_threat_agent ask --portfolio examples/portfolio.json \
  "Check NVDA and PFE for risk catalysts from the last 7 days."
```

`ask` uses the same search and validation path as `monitor`, but it is intended for ad hoc local questions.

## 7. Long-Lived Monitor

```bash
python -m portfolio_threat_agent run-monitor --portfolio examples/portfolio.json
```

The monitor:

- creates an agent once at startup
- rereads config between ticks
- supports runtime interval and on/off changes
- only sends scheduled Telegram alerts when validated new threats exist
- dedupes scheduled alerts by source URL

By default, config and state files are:

```text
state/monitor_config.json
state/monitor_state.json
```

## 8. Local Monitor Commands

```bash
python -m portfolio_threat_agent config "/change_interval 30 mins"
python -m portfolio_threat_agent config "/set_monitoring True"
python -m portfolio_threat_agent config "/set_monitoring False"
python -m portfolio_threat_agent config "/check_now"
python -m portfolio_threat_agent config "/status"
```

`/check_now` always returns a current update. If no sourced, validated threats are found, it says so.

## 9. Telegram Setup

1. In Telegram, message BotFather.
2. Create a bot.
3. Copy the bot token.
4. Set:

```bash
TELEGRAM_BOT_TOKEN="..."
```

5. Start polling:

```bash
python -m portfolio_threat_agent telegram-bot
```

You can send:

```text
/change_interval 30 mins
/set_monitoring True
/set_monitoring False
/check_now
/status
```

You can also send a portfolio JSON file/message through Telegram. The bot validates the JSON, stores it under `state/portfolios/`, records the Telegram chat ID, and updates monitor config with the local portfolio path.

Telegram polling retries transient network errors. The socket timeout is longer than Telegram's long-poll hold, so an idle poll should not kill the process.

## 10. MongoDB Persistence

Set `MONGODB_URI` to move monitor config/state from local JSON into MongoDB:

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DATABASE="portfolio_threat_agent"   # optional
export MONGODB_COLLECTION="monitor"                # optional
export MONGODB_MONITOR_ID="default"                # optional shared id
export MONGODB_MONITOR_CONFIG_ID="config:custom"   # optional explicit document id
export MONGODB_MONITOR_STATE_ID="state:custom"     # optional explicit document id
```

Mongo stores:

- `MonitorConfig`: enabled/disabled state, interval, since window, instruction, portfolio path, Telegram chat ID, model, and max results
- `MonitorState`: `alerted_source_urls` for scheduled-alert dedupe

Default document IDs:

```text
config:default
state:default
```

With `MONGODB_MONITOR_ID=prod`, those become:

```text
config:prod
state:prod
```

Important: when `MONGODB_URI` is set, Mongo is authoritative for monitor config/state. Local JSON can still exist, but it is not the active store. Telegram-uploaded portfolio payloads remain local files; Mongo stores their path.

## 11. Agent Evaluation

```bash
python -m portfolio_threat_agent eval --dataset examples/eval_cases.json
```

Evaluation is a historical agent test harness:

- each case supplies a portfolio, `start_date`, `end_date`, and expected events
- the agent is not told it is being graded
- Tavily retrieval is bounded to the case window
- Yahoo Finance enrichment uses prices at or before the historical active date
- grading reads `structured_brief.signals`, not rendered prose
- expected events must match `ticker`, `event_type`, `catalyst_date`, and materiality
- unexpected `HIGH`/`MED` signals fail; unexpected `LOW` signals are reported but allowed

## 12. LangSmith Tracing

Set:

```bash
LANGSMITH_API_KEY="lsv2_..."
LANGSMITH_PROJECT="portfolio-threat-agent"
```

Traced steps include:

- prompt construction
- agent invocation
- Tavily tool calls
- candidate validation
- Yahoo Finance enrichment
- structured brief assembly
- materiality ranking
- agent evaluation cases
- Telegram delivery

## 13. Run Tests (Coding Agent generated)

```bash
env PYTHON_DOTENV_DISABLED=true OPENAI_API_KEY=test-key TAVILY_API_KEY=test-key \
  PYTHONDONTWRITEBYTECODE=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  venv/bin/python -m unittest discover -s tests
```

The dummy keys are enough for unit tests because the suite constructs wrappers but does not make live OpenAI or Tavily calls.

The tests cover:

- agent tool setup and search budgeting
- prompt constraints
- candidate validation behavior
- Yahoo Finance enrichment paths
- materiality thresholds
- structured eval grading
- Telegram command handling and polling resilience
- Mongo/local store behavior
- rendered brief shape

## 14. Troubleshooting

### Missing `OPENAI_API_KEY`

Set `OPENAI_API_KEY` in `.env` or your shell. The project does not fall back to another model provider.

### No Tavily Results

Check:

- `TAVILY_API_KEY` is set
- the date window is not too narrow
- the holding has relevant news in the selected window

If the agent never calls Tavily, the app returns:

```text
No answer produced: agent did not search Tavily.
```

### Unknown Book Weight Or Stop Distance

This means Yahoo Finance enrichment did not return a usable quote for that ticker. The brief still renders the signal, but weight and stop context are shown as unavailable.

### Telegram Alerts Not Sending

Check:

- `TELEGRAM_BOT_TOKEN` is set
- `TELEGRAM_CHAT_ID` is set or learned from a Telegram message
- the active config store is the one you expect, especially if `MONGODB_URI` is set
- the latest scheduled run found source URLs that were not already in `alerted_source_urls`

### Agent Evaluation Fails With Missing Events

Inspect:

- the rendered brief in the eval result
- `citation_issues`
- `agent_error`
- whether Tavily retrieved sources inside the historical window
- whether the extracted event date matches the gold `catalyst_date`
