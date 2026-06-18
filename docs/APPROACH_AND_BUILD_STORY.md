# Approach And Build Story

This document explains the thinking behind the project: why this product shape was chosen, why the architecture ended up where it did, and what technical/business value it creates. It is intentionally different from [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md), which describes the implemented system in detail.

## Technical Statement

The project started as a small but useful product as a portfolio-specific threat monitor. A portfolio holder does not need every headline that mentions a ticker. They need to know whether a recent, source-backed event could matter to their actual book.

The final design separates the work into two instincts:

```text
Agent: discover candidate threats.
Application: verify, enrich, rank, render, and deliver.
```

That split is the core engineering choice. It keeps the agent useful without letting unverified agent prose become the product.

## Product Bet

Most finance news surfaces answer:

```text
What is happening in the market?
```

This project answers:

```text
What happened that could matter to this portfolio?
```

That is a different product. It requires source retrieval, holding context, exposure, stop distance, sector concentration, and a clear explanation of why a signal is material.

The value is not just summarization. The value is prioritization.

## Why The Scope Stayed Small

The project could have grown into a full trading platform, a multi-user backend, a portfolio analytics dashboard, or a broad news agent. Those would have added surface area without proving the central idea.

The useful v1 was smaller:

- portfolio JSON in
- agent-planned Tavily search
- source-backed candidate validation
- Yahoo Finance enrichment
- materiality ranking
- Telegram monitor controls
- LangSmith traces
- historical agent evaluation

That scope is enough to show the real workflow while keeping the code inspectable.

## Agentic Retrieval

An important correction during the build was rejecting fixed deterministic searches as the whole product. If the app always searches the same phrase, the LLM is mostly a summarizer. That is not a meaningful agent.

The better design is that the app provides boundaries and context, while the agent decides how to investigate each holding:

- ticker-specific searches
- company-name searches
- sector/theme searches
- follow-up searches when a result looks promising
- Tavily topic/depth/date parameters when useful

The app still enforces a hard search budget so the agent cannot spiral into expensive query loops. This is the practical compromise: agentic search planning, bounded by production controls.

## Why Tavily Is Used This Way

Tavily is the source retrieval layer, not the answer layer.

The implementation disables Tavily-generated answers so the agent and validator work from returned source results. Live runs also skip raw page Markdown to keep token use controlled. That was a deliberate simplification after traces showed that large raw payloads made runs slow and expensive.

The goal is not to feed the model everything. The goal is to feed it enough reliable source text to decide whether there is a concrete negative catalyst.

## Validation Philosophy

The validation design changed as the system got sharper.

The first instinct was a multi-stage validation flow: source parsing, faithfulness judging, threatness judging, citation contracts, and then brief assembly. That was defensible, but it became too heavy for a lightweight monitor.

The current design is simpler:

- the agent proposes a candidate threat and source URL
- the validation tool finds that URL in the current Tavily results
- one judge decides whether the source supports the claim and whether the claim is actually negative threat news
- only approved candidates are allowed into the brief

This gives one approval result instead of multiple overlapping gates. It is faster, easier to reason about, and closer to the product intent.

## Why Market Data Is App-Side

The agent searches for news. The application fetches market data after a candidate has been validated.

That choice avoids unnecessary quote calls for every search result and keeps the agent focused on discovery. Yahoo Finance data is used for:

- book weight
- exposure
- stop distance
- one-day price reaction context

Price reaction is not a gate. A threat can matter before the stock moves. The monitor is meant to surface source-backed risk catalysts, not only explain price moves after the fact.

## Materiality Is The Product Layer

The project becomes portfolio-aware when it ranks by materiality.

A generic headline can be severe in the abstract but irrelevant to a user's book. A modest-looking catalyst can be important if the position is large, the sector is concentrated, or the price is near a stop.

That is why the ranker uses transparent rules:

- position value as percent of book
- sector concentration by value
- stop proximity
- below-stop handling
- event recency and severity
- market reaction context

Rules were chosen over a learned ranker because this v1 needs to be explainable. If the system says `HIGH`, the user should see why.

## Messaging Interface

The product is intended to be easy and simple to access. The CLI exists because it is the simplest local development runner.

Slack was considered, then Telegram was chosen for this prototype because setup is simpler and it is safe to assume that most intended users of this product already use telegram in some sense for their stock/trading portfolio. All you need is a 

- one bot token
- long polling
- easy command handling
- easy portfolio JSON upload

Telegram supports interval changes, monitoring on/off, check-now, status, and portfolio updates. The long-running monitor rereads config between ticks, so changes apply without restarting the process.

## Persistence Choice

Local JSON is enough for a lightweight local prototype. MongoDB was added as an optional runtime store because monitor configuration and seen-source state should not be trapped in one local process if the system grows.

MongoDB stores:

- monitor config
- alerted source URLs

It does not currently store uploaded portfolio payloads. Telegram portfolio JSON still writes to local `state/portfolios/`. That keeps the persistence layer small while still solving the immediate runtime state problem.

## Observability

Agent systems are difficult to debug from final answers alone. LangSmith tracing was included early so the important questions are visible:

- What did the agent search?
- Which sources came back?
- Which candidates were proposed?
- Why did validation approve or reject them?
- Did ranking compute exposure correctly?
- Did Telegram send or skip the alert?
- Did eval use the correct historical window?

Tracing is part of the engineering value. It turns the agent from a black box into a system that can be inspected and improved.

## Evaluation Direction

The first evaluation shape was string matching. That was not enough.

The useful evaluation is historical simulation:

- choose a past portfolio/window
- run the normal agent without telling it it is being tested
- bound retrieval to that historical window
- use as-of market prices
- compare structured signals against expected event labels

The evaluator checks ticker, event type, catalyst date, and materiality. Unexpected HIGH/MED signals fail; unexpected LOW signals are allowed. This tests product behavior instead of final wording.

Historical windows are therefore an evaluation mechanism.

## Main Tradeoffs

### Agentic Search With A Hard Budget

The agent gets flexibility to choose searches, but tool calls are capped. This protects latency and cost while preserving agentic retrieval.

### Snippets Instead Of Raw Markdown

Skipping raw Markdown reduces token load and improves responsiveness. The tradeoff is that some article details may be unavailable if Tavily returns a weak snippet.

### One Combined Validator

Combining support and threatness into one validation result makes the flow faster and less confusing. The tradeoff is that there is less diagnostic separation than a multi-judge pipeline.

### App-Side Enrichment

Market data outside the agent keeps the agent smaller and cheaper. The tradeoff is that the agent cannot reason interactively over quote data during search, which is acceptable because price is context rather than discovery.


### Local First, Mongo Optional

Local JSON keeps the project easy to run. MongoDB supports more durable monitor state without turning the project into a hosted backend.

## Business Value

The system creates value by reducing alert noise and making risk signals portfolio-aware.

For a user, the useful output is:

- fewer generic headlines
- sourced negative catalysts
- clear materiality labels
- exposure and stop-distance context
- deduped scheduled alerts
- chat-based controls

For an engineering team, the value is a credible pattern for building agentic workflows:

- let the model search and reason
- keep tools narrow
- validate source grounding
- rank with transparent rules
- trace every step
- evaluate against structured expected outcomes

## Result

The final project is not a generic search demo. It is a focused portfolio threat agent: small enough to inspect, agentic enough to adapt searches, and constrained enough that final alerts are assembled from validated evidence rather than free-form model prose.
