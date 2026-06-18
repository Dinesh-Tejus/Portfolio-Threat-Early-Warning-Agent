# Portfolio Threat Agent

Portfolio Threat Agent is a local research-and-monitoring agent for a stock portfolio. You give it a portfolio JSON; it searches the web with Tavily for negative catalysts that could matter to each holding, validates candidate threats against the sources it actually retrieved, enriches approved signals with Yahoo Finance market context, and returns a concise portfolio threat brief. In monitor mode, validated new threats can be sent to Telegram.

The point is not to build another ticker news feed. The agent is meant to answer a narrower question:

```text
What happened recently that could matter to my portfolio, and why is it material to this book?
```

It is not a trading bot. It does not predict prices, recommend trades, or change stop levels.

## Docs

- [Setup Guide](docs/SETUP.md): install, environment variables, commands, Telegram, MongoDB, tests.
- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md): architecture, data flow, contracts, validation, monitor state.
- [Approach And Build Story](docs/APPROACH_AND_BUILD_STORY.md): why the system was built this way and what value the design creates.


## How It Works

At runtime, the application loops through the portfolio one holding at a time. For each holding, the LangChain agent receives the ticker, company/sector context, and retrieval window. The agent decides the Tavily searches to run and can call one validation tool for candidate threats it finds.

The agent does not get a market-data tool and does not send notifications directly. After the agent returns approved candidates, the application fetches Yahoo Finance prices, computes book weight and stop distance, ranks materiality, renders the brief, and handles Telegram delivery for monitoring runs.

This keeps the agent focused on discovery while keeping the final output behind explicit checks.

## Demo
[ Watch the Portfolio Threat Agent Demo](https://drive.google.com/file/d/1_HOfHT_nTs3YgZB0js0AFFnWz6pWxKUE/view?usp=sharing)

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Create a `.env` file or export these values:

```bash
export OPENAI_API_KEY="..."
export TAVILY_API_KEY="tvly-..."
export LANGSMITH_API_KEY="lsv2_..."          # optional tracing
export LANGSMITH_PROJECT="portfolio-threat-agent"
```

Run a one-time portfolio check:

```bash
python -m portfolio_threat_agent monitor \
  --portfolio examples/portfolio.json \
  --since 7d
```

The command streams progress as the agent works, then prints the final brief. The current code default is `gpt-4o-mini`.

## Portfolio Input

```json
{
  "positions": [
    { "ticker": "NVDA", "shares": 120, "stop_price": 158.0, "sector": "Semiconductors" },
    { "ticker": "PFE", "shares": 400, "stop_price": 24.5, "sector": "Pharmaceuticals" }
  ]
}
```

Required fields are `ticker` and `shares`. `stop_price` and `sector` are optional, but they make materiality ranking more useful.

## Telegram Monitor

Run the long-lived monitor:

```bash
python -m portfolio_threat_agent run-monitor --portfolio examples/portfolio.json
```

Run Telegram polling in another terminal:

```bash
export TELEGRAM_BOT_TOKEN="..."
python -m portfolio_threat_agent telegram-bot
```

Supported Telegram/local commands:

```text
/change_interval 30 mins
/set_monitoring True
/set_monitoring False
/check_now
/status
```

`/check_now` always returns an update. Scheduled monitoring only sends Telegram alerts when validated threats exist, and it dedupes scheduled alerts by source URL.

## MongoDB Runtime State

By default, monitor config and seen-source state live in local JSON under `state/`. If `MONGODB_URI` is set, the monitor, Telegram bot, and local config command use MongoDB instead:

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DATABASE="portfolio_threat_agent"   # optional
export MONGODB_COLLECTION="monitor"                # optional
export MONGODB_MONITOR_ID="default"                # optional
```

MongoDB stores monitor config and alerted source URLs. 

## Agent Evaluation

Historical windows of known news are used for agent evaluation. The eval harness runs the normal agent flow against a past retrieval window, uses as-of market prices, and grades structured signals against expected ticker/event/date/materiality labels:

```bash
python -m portfolio_threat_agent eval --dataset examples/eval_cases.json
```


#### Session logs: [View logs](https://drive.google.com/file/d/1HoEMI5mk2UpWrz5LDWa7x24X_6rKFu1B/view?usp=sharing)
