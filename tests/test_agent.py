from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
import tempfile
from unittest.mock import patch

from langgraph.errors import GraphRecursionError

from portfolio_threat_agent.config import (
    MongoMonitorConfigStore,
    MongoMonitorStateStore,
    MonitorConfigStore,
    MonitorStateStore,
    create_monitor_config_store,
    create_monitor_state_store,
    handle_monitor_command,
)
from portfolio_threat_agent.evaluation import EvalCase, ExpectedEvent, load_eval_cases, run_eval_case
from portfolio_threat_agent.agent import (
    SYSTEM_PROMPT,
    build_portfolio_instruction,
    create_portfolio_threat_agent,
    format_agent_stream_event,
    run_portfolio_entries,
)
from portfolio_threat_agent.monitor import PortfolioThreatMonitor
from portfolio_threat_agent.portfolio import Portfolio
from portfolio_threat_agent.source_parsing import extract_tavily_sources
from portfolio_threat_agent.telegram import dispatch_telegram_command, poll_telegram_once, run_telegram_polling, save_telegram_portfolio_json, send_telegram_message, telegram_api
from portfolio_threat_agent.structured import Claim, MarketQuote, Materiality, PriceReaction, Source, StructuredBrief, ThreatSignal
from portfolio_threat_agent.threats import build_structured_brief, enforce_citation_contract, rank_materiality, render_structured_brief
from portfolio_threat_agent.tools import (
    INCLUDE_DOMAINS,
    EXCLUDE_DOMAINS,
    CandidateValidationDecision,
    build_brief_from_validated_candidates,
    build_tavily_search_tool,
    reset_tavily_search_budget,
    start_tavily_search_budget,
    validate_candidate_threat,
    validate_candidate_threat_from_messages,
)


ACTIVE_DATE = date(2026, 6, 16)


def make_portfolio(*positions: dict) -> Portfolio:
    if not positions:
        positions = ({"ticker": "NVDA", "shares": 120, "sector": "Semiconductors"},)
    return Portfolio.model_validate({"positions": list(positions)})


def make_source(
    *,
    source_id: str = "S1",
    title: str = "NVDA downgrade",
    url: str = "https://example.com/2026/06/16/a",
    query: str = "NVDA downgrade",
    content: str = "NVDA downgrade after export controls.",
    published_date: date | None = ACTIVE_DATE,
) -> Source:
    return Source(
        source_id=source_id,
        title=title,
        url=url,
        query=query,
        content=content,
        published_date=published_date,
    )


def make_signal(
    *,
    ticker: str = "NVDA",
    event_type: str = "downgrade",
    headline: str | None = None,
    summary: str | None = None,
    claim_text: str | None = None,
    source_ids: list[str] | None = None,
    catalyst_date: date | None = ACTIVE_DATE,
    **overrides,
) -> ThreatSignal:
    source_ids = source_ids or ["S1"]
    headline = headline or f"{ticker} {event_type}"
    summary = summary or f"{ticker} {event_type}."
    claim_text = claim_text or summary
    claims = overrides.pop("claims", [Claim(text=claim_text, source_ids=source_ids)])
    return ThreatSignal(
        ticker=ticker,
        event_type=event_type,
        headline=headline,
        summary=summary,
        claims=claims,
        source_ids=source_ids,
        catalyst_date=catalyst_date,
        **overrides,
    )


class FakeMongoCollection:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}

    def find_one(self, query: dict) -> dict | None:
        document = self.documents.get(query["_id"])
        return dict(document) if document else None

    def update_one(self, query: dict, update: dict, upsert: bool = False) -> None:
        document = self.documents.get(query["_id"], {"_id": query["_id"]})
        document.update(update.get("$set", {}))
        self.documents[query["_id"]] = document


class AgentTests(unittest.TestCase):
    def test_prompt_sets_scope(self) -> None:
        prompt_lower = SYSTEM_PROMPT.lower()
        self.assertIn("use tavily as source of truth", prompt_lower)
        self.assertIn("do not give buy/sell advice", prompt_lower)
        self.assertIn("do not answer from memory", prompt_lower)
        self.assertIn("price-move explanations", prompt_lower)
        self.assertIn("bounded pass", prompt_lower)

    def test_create_agent_uses_single_tavily_tool(self) -> None:
        fake_tool = object()
        fake_model = object()
        fake_agent = object()
        with (
            patch("portfolio_threat_agent.agent.create_chat_model", return_value=fake_model) as chat_mock,
            patch("portfolio_threat_agent.tools.TavilySearch", return_value=type("FakeTavily", (), {"description": "search", "invoke": lambda self, args: {"query": args.get("query"), "results": []}})()) as tavily_mock,
            patch("portfolio_threat_agent.agent.create_agent", return_value=fake_agent) as create_mock,
        ):
            agent = create_portfolio_threat_agent(model="test-model", max_results=3)

        self.assertIs(agent, fake_agent)
        chat_mock.assert_called_once_with(model="test-model", temperature=0)
        tavily_mock.assert_called_once_with(max_results=3, auto_parameters=True, include_raw_content=False, include_answer=False, include_domains=INCLUDE_DOMAINS, exclude_domains=EXCLUDE_DOMAINS)
        search_tool = create_mock.call_args.kwargs["tools"][0]
        self.assertEqual(search_tool.name, "tavily_search")
        create_mock.assert_called_once_with(
            model=fake_model,
            tools=[search_tool, validate_candidate_threat],
            system_prompt=SYSTEM_PROMPT,
        )

    def test_tavily_tool_enforces_per_run_search_budget(self) -> None:
        calls: list[str] = []

        class FakeTavily:
            description = "search"

            def invoke(self, args: dict) -> dict:
                calls.append(str(args.get("query")))
                return {"query": args.get("query"), "results": [{"url": "https://example.com"}]}

        with patch("portfolio_threat_agent.tools.TavilySearch", return_value=FakeTavily()):
            search_tool = build_tavily_search_tool()

        token = start_tavily_search_budget()
        try:
            results = [search_tool.invoke({"query": f"q{i}"}) for i in range(5)]
        finally:
            reset_tavily_search_budget(token)

        self.assertEqual(len(calls), 4)
        self.assertNotIn("error", results[3])
        self.assertEqual(results[4]["error"], "search_budget_exhausted: max 4 Tavily searches per holding run")

    def test_build_portfolio_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portfolio.json"
            path.write_text('{"positions":[{"ticker":"PFE","shares":400,"stop_price":24.5}]}')
            prompt = build_portfolio_instruction(
                path,
                since="7d",
                extra_instruction="Focus on FDA news.",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 3),
            )
        self.assertIn("PFE", prompt)
        self.assertIn("Active date: 2024-01-03", prompt)
        self.assertIn("Retrieval start date: 2024-01-01", prompt)
        self.assertIn("Retrieval end date: 2024-01-03", prompt)
        self.assertIn("serious but bounded Tavily pass", prompt)
        self.assertIn("max three initial searches plus one focused follow-up", prompt)
        self.assertIn("ticker/company news", prompt)
        self.assertIn("price-move explanations", prompt)
        self.assertIn("sector/theme context", prompt)
        self.assertIn("Run follow-up searches", prompt)
        self.assertIn("You choose exact queries", prompt)
        self.assertIn("validate_candidate_threat", prompt)
        self.assertIn("Do not call price", prompt)
        self.assertIn("final brief", prompt)
        self.assertIn("Focus on FDA news.", prompt)
        self.assertNotIn("evaluation", prompt.lower())
        self.assertNotIn("being tested", prompt.lower())
        self.assertNotIn("virtual", prompt.lower())

    def test_telegram_message_skips_without_bot_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = send_telegram_message("hello")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "TELEGRAM_BOT_TOKEN is not configured")

    def test_telegram_message_uses_bot_token_and_chat(self) -> None:
        fake_response = {"ok": True, "result": {"message_id": 123}}
        with (
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "telegram-test"}, clear=True),
            patch("portfolio_threat_agent.telegram.telegram_api", return_value=fake_response) as api_mock,
        ):
            result = send_telegram_message("hello", chat_id="12345")

        api_mock.assert_called_once_with("sendMessage", {"chat_id": "12345", "text": "hello"})
        self.assertEqual(result["status"], "sent")

    def test_telegram_api_uses_socket_timeout(self) -> None:
        response = type("Response", (), {"read": lambda self: b'{"ok": true}'})()
        opener = type("Opener", (), {"__enter__": lambda self: response, "__exit__": lambda self, *args: None})()
        with patch("portfolio_threat_agent.telegram.request.urlopen", return_value=opener) as urlopen_mock:
            result = telegram_api("getMe", {}, token="telegram-test", socket_timeout=17)

        self.assertTrue(result["ok"])
        self.assertEqual(urlopen_mock.call_args.kwargs["timeout"], 17)

    def test_poll_telegram_once_uses_socket_timeout_above_long_poll_timeout(self) -> None:
        with patch("portfolio_threat_agent.telegram.telegram_api", return_value={"ok": True, "result": []}) as api_mock:
            offset = poll_telegram_once(object(), offset=42, timeout=30)

        self.assertEqual(offset, 42)
        self.assertEqual(api_mock.call_args.kwargs["socket_timeout"], 40)

    def test_run_telegram_polling_retries_transient_errors(self) -> None:
        monitor = object()
        with (
            patch("portfolio_threat_agent.telegram.poll_telegram_once", side_effect=[TimeoutError("timed out"), KeyboardInterrupt]) as poll_mock,
            patch("portfolio_threat_agent.telegram.time.sleep") as sleep_mock,
            patch("builtins.print") as print_mock,
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_telegram_polling(monitor)

        self.assertEqual(poll_mock.call_count, 2)
        sleep_mock.assert_called_once_with(5)
        self.assertIn("TimeoutError", print_mock.call_args.args[0])

    def test_stream_formatter_summarizes_tool_calls_and_results(self) -> None:
        ai_message = type(
            "Message",
            (),
            {"content": "", "type": "ai", "tool_calls": [{"name": "tavily_search", "args": {"query": "NVDA risk", "topic": "news"}}]},
        )()
        tool_message = type(
            "Message",
            (),
            {"content": '{"query":"NVDA risk","results":[{"url":"https://example.com"}]}', "type": "tool", "name": "tavily_search"},
        )()
        validator_tool_message = type(
            "Message",
            (),
            {
                "content": '{"status":"approved","ticker":"NVDA","reason":"Concrete downgrade"}',
                "type": "tool",
                "name": "validate_candidate_threat",
            },
        )()

        self.assertEqual(format_agent_stream_event(ai_message), "agent called tavily_search(query='NVDA risk', topic='news')")
        self.assertEqual(format_agent_stream_event(tool_message), "tavily_search returned 1 result for 'NVDA risk'")
        self.assertEqual(format_agent_stream_event(validator_tool_message), "validate_candidate_threat approved for NVDA: Concrete downgrade")

    def test_validate_candidate_threat_approves_supported_threat(self) -> None:
        tavily_message = type(
            "Message",
            (),
            {
                "content": (
                    '{"query":"NVDA downgrade","results":[{"title":"NVDA downgraded","url":"https://example.com/2026/06/16/a",'
                    '"content":"NVDA was downgraded after export control risk.","raw_content":"NVDA was downgraded after export control risk.",'
                    '"published_date":"2026-06-16"}]}'
                ),
                "type": "tool",
                "name": "tavily_search",
            },
        )()

        result = validate_candidate_threat_from_messages(
            ticker="NVDA",
            event_type="downgrade",
            claim="NVDA was downgraded after export control risk.",
            source_url="https://example.com/2026/06/16/a",
            active_date="2026-06-17",
            messages=[tavily_message],
            candidate_judge=lambda ticker, event_type, claim, excerpt: CandidateValidationDecision(verdict="approved", reason="Concrete downgrade."),
        )

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["ticker"], "NVDA")
        self.assertEqual(result["event_type"], "downgrade")
        self.assertEqual(result["source_url"], "https://example.com/2026/06/16/a")

    def test_validate_candidate_threat_allows_undated_tavily_source(self) -> None:
        tavily_message = type(
            "Message",
            (),
            {
                "content": (
                    '{"query":"TSLA recall","results":[{"title":"TSLA recall risk","url":"https://example.com/tsla-recall",'
                    '"content":"TSLA faces a recall risk.","raw_content":"TSLA faces a recall risk."}]}'
                ),
                "type": "tool",
                "name": "tavily_search",
            },
        )()

        result = validate_candidate_threat_from_messages(
            ticker="TSLA",
            event_type="regulatory",
            claim="TSLA faces a recall risk.",
            source_url="https://example.com/tsla-recall",
            active_date="2026-06-17",
            messages=[tavily_message],
            candidate_judge=lambda ticker, event_type, claim, excerpt: CandidateValidationDecision(verdict="approved", reason="Concrete recall risk."),
        )

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["catalyst_date"], "2026-06-17")

    def test_validate_candidate_threat_short_circuits_duplicate_source_event(self) -> None:
        tavily_message = type(
            "Message",
            (),
            {
                "content": (
                    '{"query":"ADSK stock down","results":[{"title":"ADSK drops","url":"https://example.com/adsk",'
                    '"content":"ADSK dropped with software stocks.","raw_content":"ADSK dropped with software stocks."}]}'
                ),
                "type": "tool",
                "name": "tavily_search",
            },
        )()
        prior_validation = type(
            "Message",
            (),
            {
                "content": (
                    '{"status":"rejected","reason":"claim is unsupported by cited source","ticker":"ADSK",'
                    '"event_type":"sector downturn","claim":"ADSK dropped 2.5%.","source_url":"https://example.com/adsk"}'
                ),
                "type": "tool",
                "name": "validate_candidate_threat",
            },
        )()

        def fail_judge(ticker: str, event_type: str, claim: str, excerpt: str) -> CandidateValidationDecision:
            raise AssertionError("duplicate validation should not call the judge")

        result = validate_candidate_threat_from_messages(
            ticker="ADSK",
            event_type="Sector Downturn",
            claim="Autodesk dropped with software peers.",
            source_url="https://example.com/adsk",
            active_date="2026-06-17",
            messages=[tavily_message, prior_validation],
            candidate_judge=fail_judge,
        )

        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(result["previous_status"], "rejected")
        self.assertIn("already validated", result["reason"])

    def test_validate_candidate_threat_ignores_malformed_prior_tool_json(self) -> None:
        tavily_message = type(
            "Message",
            (),
            {
                "content": (
                    '{"query":"ADSK stock down","results":[{"title":"ADSK drops","url":"https://example.com/adsk",'
                    '"content":"ADSK dropped with software stocks.","raw_content":"ADSK dropped with software stocks."}]}'
                ),
                "type": "tool",
                "name": "tavily_search",
            },
        )()
        malformed_prior_validation = type(
            "Message",
            (),
            {
                "content": "Error: tool call failed with payload {'not': 'json'} and trailing text",
                "type": "tool",
                "name": "validate_candidate_threat",
            },
        )()

        result = validate_candidate_threat_from_messages(
            ticker="ADSK",
            event_type="Sector Downturn",
            claim="ADSK dropped with software stocks.",
            source_url="https://example.com/adsk",
            active_date="2026-06-17",
            messages=[tavily_message, malformed_prior_validation],
            candidate_judge=lambda ticker, event_type, claim, excerpt: CandidateValidationDecision(verdict="approved", reason="Negative sector move."),
        )

        self.assertEqual(result["status"], "approved")

    def test_validate_candidate_threat_rejects_non_threat_even_when_supported(self) -> None:
        tavily_message = type(
            "Message",
            (),
            {
                "content": (
                    '{"query":"NVDA risk","results":[{"title":"NVDA rises","url":"https://example.com/2026/06/16/a",'
                    '"content":"NVDA shares rose after bullish analyst commentary.","raw_content":"NVDA shares rose after bullish analyst commentary.",'
                    '"published_date":"2026-06-16"}]}'
                ),
                "type": "tool",
                "name": "tavily_search",
            },
        )()

        result = validate_candidate_threat_from_messages(
            ticker="NVDA",
            event_type="other",
            claim="NVDA shares rose after bullish analyst commentary.",
            source_url="https://example.com/2026/06/16/a",
            active_date="2026-06-17",
            messages=[tavily_message],
            candidate_judge=lambda ticker, event_type, claim, excerpt: CandidateValidationDecision(verdict="not_threat", reason="Bullish news."),
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "candidate is not concrete negative threat news")
        self.assertEqual(result["feedback"], "Bullish news.")

    def test_validate_candidate_threat_rejects_unsupported_claim(self) -> None:
        tavily_message = type(
            "Message",
            (),
            {
                "content": (
                    '{"query":"NVDA risk","results":[{"title":"NVDA article","url":"https://example.com/2026/06/16/a",'
                    '"content":"NVDA shares rose after bullish analyst commentary.","raw_content":"NVDA shares rose after bullish analyst commentary.",'
                    '"published_date":"2026-06-16"}]}'
                ),
                "type": "tool",
                "name": "tavily_search",
            },
        )()

        result = validate_candidate_threat_from_messages(
            ticker="NVDA",
            event_type="downgrade",
            claim="NVDA was downgraded after export controls.",
            source_url="https://example.com/2026/06/16/a",
            active_date="2026-06-17",
            messages=[tavily_message],
            candidate_judge=lambda ticker, event_type, claim, excerpt: CandidateValidationDecision(verdict="unsupported", reason="The source does not say this."),
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "claim is unsupported by cited source")
        self.assertEqual(result["feedback"], "The source does not say this.")

    def test_validate_candidate_threat_rejects_source_not_from_tavily(self) -> None:
        result = validate_candidate_threat_from_messages(
            ticker="NVDA",
            event_type="downgrade",
            claim="NVDA was downgraded.",
            source_url="https://example.com/missing",
            active_date="2026-06-17",
            messages=[],
            candidate_judge=lambda ticker, event_type, claim, excerpt: CandidateValidationDecision(verdict="approved"),
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "source_url was not found in Tavily results for this agent run")

    def test_run_portfolio_entries_reuses_one_agent_with_per_position_prompts(self) -> None:
        portfolio = make_portfolio(
            {"ticker": "NVDA", "shares": 1, "sector": "Semiconductors"},
            {"ticker": "PFE", "shares": 1, "sector": "Pharma"},
        )
        fake_agent = object()
        prompts: list[str] = []

        def fake_run(agent: object, prompt: str, stream_handler=None):
            prompts.append(prompt)
            return [
                type(
                    "Message",
                    (),
                    {
                        "content": '{"query":"threat check","results":[]}',
                        "type": "tool",
                        "name": "tavily_search",
                    },
                )()
            ]

        with patch("portfolio_threat_agent.agent.run_agent_messages", side_effect=fake_run) as run_mock:
            _messages, results = run_portfolio_entries(
                fake_agent,
                portfolio,
                since="4d",
                instruction=None,
                active_date=ACTIVE_DATE,
                retrieval_start=ACTIVE_DATE,
            )

        self.assertEqual(run_mock.call_count, 2)
        self.assertTrue(all(call.args[0] is fake_agent for call in run_mock.call_args_list))
        self.assertIn("NVDA", prompts[0])
        self.assertNotIn("PFE", prompts[0])
        self.assertIn("PFE", prompts[1])
        self.assertNotIn("NVDA", prompts[1])
        self.assertEqual(len(results), 2)

    def test_validated_brief_uses_full_portfolio_for_book_weight(self) -> None:
        active = date.today()
        portfolio = make_portfolio(
            {"ticker": "NVDA", "shares": 10, "stop_price": 80, "sector": "Semiconductors"},
            {"ticker": "TSLA", "shares": 10, "stop_price": 45, "sector": "Autos"},
        )
        tavily_message = type(
            "Message",
            (),
            {
                "content": (
                    f'{{"query":"TSLA recall","results":[{{"title":"TSLA recall","url":"https://example.com/{active}/tsla",'
                    f'"content":"TSLA faces a recall risk.","raw_content":"TSLA faces a recall risk.","published_date":"{active}"}}]}}'
                ),
                "type": "tool",
                "name": "tavily_search",
            },
        )()
        validator_message = type(
            "Message",
            (),
            {
                "content": (
                    f'{{"status":"approved","ticker":"TSLA","event_type":"regulatory","claim":"TSLA faces a recall risk.",'
                    f'"headline":"TSLA recall","source_url":"https://example.com/{active}/tsla","source_id":"S1",'
                    f'"catalyst_date":"{active}","reason":"Concrete recall risk."}}'
                ),
                "type": "tool",
                "name": "validate_candidate_threat",
            },
        )()

        def quote(ticker: str) -> MarketQuote:
            prices = {"NVDA": 100.0, "TSLA": 50.0}
            return MarketQuote(ticker=ticker, price=prices[ticker], currency="USD")

        reaction = PriceReaction(
            ticker="TSLA",
            catalyst_date=active,
            active_date=active,
            start_price=52.0,
            end_price=50.0,
            pct_change=-3.8,
            currency="USD",
        )
        with (
            patch("portfolio_threat_agent.tools.get_yfinance_quote", side_effect=quote),
            patch("portfolio_threat_agent.tools.get_yfinance_price_reaction", return_value=reaction),
        ):
            result = build_brief_from_validated_candidates(portfolio, active, [tavily_message, validator_message])

        signal = result["structured_brief"]["signals"][0]
        self.assertAlmostEqual(signal["portfolio_weight"], 1 / 3)
        self.assertIn("33.3% of book", result["rendered_brief"])
        self.assertNotIn("100.0% of book", result["rendered_brief"])

    def test_monitor_commands_update_config_and_run_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MonitorConfigStore(Path(tmpdir) / "config.json")
            interval_response = handle_monitor_command("/change_interval 30 minutes", store=store)
            disable_response = handle_monitor_command("/set_monitoring False", store=store)
            check_response = handle_monitor_command("/check_now", store=store, check_now=lambda: "checked")
            config = store.load()
        self.assertIn("30 minutes", interval_response)
        self.assertEqual(config.interval_seconds, 1800)
        self.assertEqual(disable_response, "Monitoring disabled.")
        self.assertFalse(config.monitoring_enabled)
        self.assertEqual(check_response, "checked")

    def test_mongo_monitor_config_store_persists_config(self) -> None:
        collection = FakeMongoCollection()
        store = MongoMonitorConfigStore(collection_obj=collection, document_id="config:test")
        config = store.set_interval(1800)
        config = config.model_copy(update={"portfolio_path": "examples/portfolio.json", "telegram_chat_id": "123"})
        store.save(config)

        reloaded = MongoMonitorConfigStore(collection_obj=collection, document_id="config:test").load()

        self.assertEqual(reloaded.interval_seconds, 1800)
        self.assertEqual(reloaded.portfolio_path, "examples/portfolio.json")
        self.assertEqual(reloaded.telegram_chat_id, "123")

    def test_mongo_monitor_state_store_remembers_source_urls(self) -> None:
        collection = FakeMongoCollection()
        store = MongoMonitorStateStore(collection_obj=collection, document_id="state:test")

        store.remember_source_urls({"https://example.com/a"})
        state = store.remember_source_urls({"https://example.com/a", "https://example.com/b"})

        self.assertEqual(state.alerted_source_urls, ["https://example.com/a", "https://example.com/b"])

    def test_monitor_store_factory_uses_mongo_when_uri_is_configured(self) -> None:
        with patch.dict("os.environ", {"MONGODB_URI": "mongodb://localhost:27017"}, clear=True):
            with (
                patch("portfolio_threat_agent.config.MongoMonitorConfigStore", return_value="config-store"),
                patch("portfolio_threat_agent.config.MongoMonitorStateStore", return_value="state-store"),
            ):
                self.assertEqual(create_monitor_config_store(), "config-store")
                self.assertEqual(create_monitor_state_store(), "state-store")

    def test_mongo_monitor_id_namespaces_config_and_state_documents(self) -> None:
        with patch.dict("os.environ", {"MONGODB_MONITOR_ID": "user-1"}, clear=True):
            config_store = MongoMonitorConfigStore(collection_obj=FakeMongoCollection())
            state_store = MongoMonitorStateStore(collection_obj=FakeMongoCollection())

        self.assertEqual(config_store.document_id, "config:user-1")
        self.assertEqual(state_store.document_id, "state:user-1")

    def test_monitor_respects_disabled_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            portfolio_path = Path(tmpdir) / "portfolio.json"
            portfolio_path.write_text('{"positions":[{"ticker":"NVDA","shares":1}]}')
            store = MonitorConfigStore(config_path)
            config = store.load().model_copy(update={"portfolio_path": str(portfolio_path), "monitoring_enabled": False})
            store.save(config)

            with patch("portfolio_threat_agent.monitor.run_portfolio_entries") as run_mock:
                monitor = PortfolioThreatMonitor(store)
                response = monitor.run_once()

        self.assertEqual(response, "Monitoring is disabled.")
        run_mock.assert_not_called()

    def test_check_now_returns_current_brief_and_lets_agent_decide_notification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            state_path = Path(tmpdir) / "state.json"
            portfolio_path = Path(tmpdir) / "portfolio.json"
            portfolio_path.write_text('{"positions":[{"ticker":"NVDA","shares":1}]}')
            store = MonitorConfigStore(config_path)
            state_store = MonitorStateStore(state_path)
            config = store.load().model_copy(update={"portfolio_path": str(portfolio_path)})
            store.save(config)
            validated_result = {
                "status": "ok",
                "rendered_brief": "LOW · NVDA threat",
                "signal_count": 1,
                "source_urls": ["https://example.com/nvda-threat"],
                "blocked_issues": [],
            }

            with (
                patch("portfolio_threat_agent.monitor.PortfolioThreatMonitor.start_monitor"),
                patch("portfolio_threat_agent.monitor.create_portfolio_threat_agent", return_value=object()),
                patch("portfolio_threat_agent.monitor.run_portfolio_entries", return_value=([], [validated_result])),
            ):
                monitor = PortfolioThreatMonitor(store, state_store=state_store)
                response = monitor.run_once(force=True)
                seen_urls = state_store.load().alerted_source_urls

        self.assertIn("NVDA", response)
        self.assertIn("https://example.com/nvda-threat", seen_urls)

    def test_scheduled_monitor_sends_only_for_new_source_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            state_path = Path(tmpdir) / "state.json"
            portfolio_path = Path(tmpdir) / "portfolio.json"
            portfolio_path.write_text('{"positions":[{"ticker":"NVDA","shares":1}]}')
            store = MonitorConfigStore(config_path)
            state_store = MonitorStateStore(state_path)
            config = store.load().model_copy(update={"portfolio_path": str(portfolio_path)})
            store.save(config)
            validated_result = {
                "status": "ok",
                "rendered_brief": "LOW · NVDA threat",
                "signal_count": 1,
                "source_urls": ["https://example.com/nvda-threat"],
                "blocked_issues": [],
            }

            with (
                patch("portfolio_threat_agent.monitor.PortfolioThreatMonitor.start_monitor"),
                patch("portfolio_threat_agent.monitor.create_portfolio_threat_agent", return_value=object()),
                patch("portfolio_threat_agent.monitor.run_portfolio_entries", side_effect=[([], [validated_result]), ([], [validated_result])]) as run_mock,
                patch("portfolio_threat_agent.monitor.send_telegram_message", return_value={"status": "sent"}) as send_mock,
            ):
                monitor = PortfolioThreatMonitor(store, state_store=state_store)
                first = monitor.run_once()
                second = monitor.run_once()

        self.assertIn("NVDA", first)
        self.assertIn("NVDA", second)
        self.assertEqual(run_mock.call_count, 2)
        send_mock.assert_called_once_with("LOW · NVDA threat")

    def test_extract_tavily_sources_from_tool_message(self) -> None:
        content = (
            '{"query":"NVDA downgrade","results":[{"title":"NVDA risk","url":"https://example.com/2026/06/16/a",'
            '"content":"NVDA was downgraded.","raw_content":"NVDA was downgraded.","score":0.9}]}'
        )
        tool_message = type("Message", (), {"content": content, "type": "tool", "name": "tavily_search"})()
        sources = extract_tavily_sources([tool_message])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].published_date.isoformat(), "2026-06-16")

    def test_materiality_treats_below_stop_as_high_proximity(self) -> None:
        portfolio = make_portfolio({"ticker": "NVDA", "shares": 10, "stop_price": 100, "sector": "Semiconductors"})
        signal = make_signal(summary="NVDA was downgraded.")
        ranked = rank_materiality(
            portfolio,
            [signal],
            active_date=ACTIVE_DATE,
            quotes={"NVDA": MarketQuote(ticker="NVDA", price=95, currency="USD")},
        )

        self.assertAlmostEqual(ranked[0].distance_to_stop_pct or 0, -5.0, places=2)
        self.assertEqual(ranked[0].materiality.value, "HIGH")
        self.assertIn("below stop", ranked[0].materiality_reason)

    def test_materiality_high_uses_prd_within_10_percent_stop_threshold(self) -> None:
        portfolio = make_portfolio({"ticker": "NVDA", "shares": 10, "stop_price": 100, "sector": "Semiconductors"})
        signal = make_signal(summary="NVDA was downgraded.")
        ranked = rank_materiality(
            portfolio,
            [signal],
            active_date=ACTIVE_DATE,
            quotes={"NVDA": MarketQuote(ticker="NVDA", price=109, currency="USD")},
        )

        self.assertEqual(ranked[0].materiality, Materiality.HIGH)
        self.assertIn("within 10% of stop", ranked[0].materiality_reason)

    def test_materiality_medium_uses_prd_within_20_percent_stop_threshold(self) -> None:
        portfolio = make_portfolio(
            {"ticker": "A", "shares": 100, "stop_price": 1, "sector": "One"},
            {"ticker": "B", "shares": 90, "stop_price": 1, "sector": "Two"},
            {"ticker": "C", "shares": 80, "stop_price": 1, "sector": "Three"},
            {"ticker": "D", "shares": 70, "stop_price": 1, "sector": "Four"},
            {"ticker": "E", "shares": 60, "stop_price": 1, "sector": "Five"},
            {"ticker": "NVDA", "shares": 1, "stop_price": 100, "sector": "Semiconductors"},
        )
        signal = make_signal(summary="NVDA was downgraded.")
        ranked = rank_materiality(
            portfolio,
            [signal],
            active_date=ACTIVE_DATE,
            quotes={
                "A": MarketQuote(ticker="A", price=100, currency="USD"),
                "B": MarketQuote(ticker="B", price=100, currency="USD"),
                "C": MarketQuote(ticker="C", price=100, currency="USD"),
                "D": MarketQuote(ticker="D", price=100, currency="USD"),
                "E": MarketQuote(ticker="E", price=100, currency="USD"),
                "NVDA": MarketQuote(ticker="NVDA", price=119, currency="USD"),
            },
        )

        self.assertEqual(ranked[0].materiality, Materiality.MED)
        self.assertIn("within 20% of stop", ranked[0].materiality_reason)

    def test_materiality_ranks_by_quoted_position_value_not_share_count(self) -> None:
        portfolio = make_portfolio(
            {"ticker": "NVDA", "shares": 120, "sector": "Semiconductors"},
            {"ticker": "PFE", "shares": 400, "sector": "Health Care"},
        )
        signals = [
            make_signal(ticker="PFE", event_type="regulatory", source_ids=["S1"]),
            make_signal(ticker="NVDA", event_type="regulatory", source_ids=["S2"]),
        ]
        ranked = rank_materiality(
            portfolio,
            signals,
            active_date=ACTIVE_DATE,
            quotes={
                "NVDA": MarketQuote(ticker="NVDA", price=200, currency="USD"),
                "PFE": MarketQuote(ticker="PFE", price=25, currency="USD"),
            },
        )

        self.assertEqual(ranked[0].ticker, "NVDA")
        self.assertGreater(ranked[0].position_value or 0, ranked[1].position_value or 0)
        self.assertIn("quoted position value", ranked[0].materiality_reason)

    def test_sector_concentration_uses_quoted_book_value_not_position_count(self) -> None:
        portfolio = make_portfolio(
            {"ticker": "NVDA", "shares": 100, "sector": "Semiconductors"},
            {"ticker": "PFE", "shares": 10, "sector": "Health Care"},
            {"ticker": "TSLA", "shares": 10, "sector": "Consumer Discretionary"},
            {"ticker": "F", "shares": 10, "sector": "Consumer Discretionary"},
        )
        signal = make_signal(event_type="regulatory")

        ranked = rank_materiality(
            portfolio,
            [signal],
            active_date=ACTIVE_DATE,
            quotes={
                "NVDA": MarketQuote(ticker="NVDA", price=100, currency="USD"),
                "PFE": MarketQuote(ticker="PFE", price=10, currency="USD"),
                "TSLA": MarketQuote(ticker="TSLA", price=10, currency="USD"),
                "F": MarketQuote(ticker="F", price=10, currency="USD"),
            },
        )

        self.assertIn("overweight sector/theme by quoted book value", ranked[0].materiality_reason)

    def test_citation_contract_allows_undated_source_from_bounded_tavily_search(self) -> None:
        source = make_source(url="https://example.com/a", query="NVDA", content="NVDA downgrade", published_date=None)
        signal = make_signal(summary="NVDA was downgraded.", catalyst_date=None)
        valid, issues = enforce_citation_contract([signal], [source], active_date=ACTIVE_DATE)
        self.assertEqual(valid, [signal])
        self.assertEqual(issues, [])

    def test_citation_contract_does_not_date_gate_when_search_window_is_bounded(self) -> None:
        source = make_source(published_date=date(2026, 6, 18))
        signal = make_signal(summary="NVDA was downgraded.")
        valid, issues = enforce_citation_contract([signal], [source], active_date=ACTIVE_DATE)
        self.assertEqual(valid, [signal])
        self.assertEqual(issues, [])

    def test_citation_contract_blocks_unknown_source_id(self) -> None:
        signal = make_signal(summary="NVDA was downgraded.", catalyst_date=None)
        valid, issues = enforce_citation_contract([signal], [])
        self.assertEqual(valid, [])
        self.assertTrue(any("unknown" in issue.issue for issue in issues))

    def test_structured_brief_uses_explicit_validated_signals(self) -> None:
        portfolio = make_portfolio()
        source = make_source(
            title="NVDA lawsuit",
            query="NVDA lawsuit",
            content="NVDA was sued over alleged disclosure issues.",
        )
        signal = make_signal(
            event_type="lawsuit",
            headline="NVDA lawsuit",
            summary="NVDA was sued over alleged disclosure issues.",
            claim_text="NVDA was sued over alleged disclosure issues.",
        )

        brief = build_structured_brief(portfolio, [source], [signal], active_date=ACTIVE_DATE)
        self.assertEqual(brief.signals[0].event_type, "lawsuit")

    def test_render_structured_brief_matches_prd_citation_shape(self) -> None:
        sources = [
            Source(
                source_id="S1",
                title="Reuters NVDA downgrade",
                url="https://example.com/reuters",
                query="NVDA downgrade",
                content="NVDA was downgraded.",
                published_date=ACTIVE_DATE,
            ),
            Source(
                source_id="S2",
                title="Bloomberg NVDA target cut",
                url="https://example.com/bloomberg",
                query="NVDA price target cut",
                content="Analysts cut targets.",
                published_date=ACTIVE_DATE,
            ),
        ]
        signal = make_signal(
            headline="NVDA downgrade",
            summary="NVDA was downgraded.",
            claims=[
                Claim(text="NVDA was downgraded by analysts.", source_ids=["S1"]),
                Claim(text="Analysts cut NVDA price targets.", source_ids=["S1", "S2"]),
            ],
            source_ids=["S1", "S2"],
            materiality=Materiality.HIGH,
            materiality_reason="top-3 position by quoted position value + within 10% of stop",
            position_value=41000,
            portfolio_weight=0.082,
            distance_to_stop_pct=4.1,
        )
        brief = StructuredBrief(active_date=ACTIVE_DATE, sources=sources, signals=[signal])

        rendered = render_structured_brief(brief)

        self.assertIn("HIGH · NVDA (8.2% of book · 4.1% above stop)", rendered)
        self.assertIn("NVDA was downgraded by analysts. [1]", rendered)
        self.assertIn("Analysts cut NVDA price targets. [1][2]", rendered)
        self.assertIn("Exposure: $41,000 · Distance to stop: 4.1% above stop · Why HIGH:", rendered)
        self.assertIn("[1] Reuters NVDA downgrade, 2026-06-16 https://example.com/reuters", rendered)
        self.assertIn("[2] Bloomberg NVDA target cut, 2026-06-16 https://example.com/bloomberg", rendered)

    def test_telegram_command_dispatch_uses_monitor(self) -> None:
        monitor = type("Monitor", (), {"handle_command": lambda self, text, chat_id=None: f"handled {text}"})()
        monitor.config_store = type("Store", (), {"set_telegram_chat": lambda self, chat_id: None})()
        response = dispatch_telegram_command("/change_interval 30 mins", "12345", monitor)
        self.assertEqual(response, "handled /change_interval 30 mins")

    def test_save_telegram_portfolio_json_updates_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MonitorConfigStore(Path(tmpdir) / "config.json")
            response = save_telegram_portfolio_json(
                '{"positions":[{"ticker":"NVDA","shares":1}]}',
                "12345",
                "67890",
                store,
                root=Path(tmpdir) / "portfolios",
            )
            config = store.load()
        self.assertIn("Portfolio saved", response)
        self.assertIsNotNone(config.portfolio_path)
        self.assertEqual(config.telegram_chat_id, "12345")

    def test_eval_dataset_loads_golden_event_metadata(self) -> None:
        cases = load_eval_cases(Path("examples/eval_cases.json"))
        self.assertGreater(len(cases), 0)
        for case in cases:
            self.assertTrue(case.id)
            self.assertIsNotNone(case.start_date)
            self.assertIsNotNone(case.end_date)
            self.assertGreater(len(case.expected_events), 0)
            for event in case.expected_events:
                self.assertTrue(event.ticker)
                self.assertIsNotNone(event.expected_materiality)

    def test_eval_case_grades_structured_events_not_strings(self) -> None:
        case = EvalCase(
            id="unit",
            portfolio_path="examples/golden_portfolio.json",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            expected_events=[
                ExpectedEvent(
                    ticker="AAPL",
                    event_type="downgrade",
                    catalyst_date=date(2024, 1, 2),
                    expected_materiality=Materiality.HIGH,
                )
            ],
        )
        validated_result = {
            "status": "ok",
            "rendered_brief": "HIGH · AAPL",
            "signal_count": 1,
            "blocked_issues": [],
            "structured_brief": {
                "signals": [
                    {
                        "ticker": "AAPL",
                        "event_type": "downgrade",
                        "catalyst_date": "2024-01-02",
                        "materiality": "HIGH",
                        "headline": "Apple downgraded",
                        "source_urls": ["https://example.com/aapl"],
                    }
                ]
            },
        }

        with (
            patch("portfolio_threat_agent.evaluation.create_portfolio_threat_agent", return_value=object()),
            patch("portfolio_threat_agent.evaluation.run_portfolio_entries", return_value=([], [validated_result])),
        ):
            result = run_eval_case(case)

        self.assertTrue(result.passed)
        self.assertEqual(result.matched_events[0].ticker, "AAPL")

    def test_eval_case_records_agent_loop_failure(self) -> None:
        case = EvalCase(
            id="loop",
            portfolio_path="examples/golden_portfolio.json",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            expected_events=[
                ExpectedEvent(
                    ticker="AAPL",
                    event_type="downgrade",
                    catalyst_date=date(2024, 1, 2),
                    expected_materiality=Materiality.HIGH,
                )
            ],
        )

        with (
            patch("portfolio_threat_agent.evaluation.create_portfolio_threat_agent", return_value=object()),
            patch("portfolio_threat_agent.evaluation.run_portfolio_entries", side_effect=GraphRecursionError("limit reached")),
        ):
            result = run_eval_case(case)

        self.assertFalse(result.passed)
        self.assertEqual(result.agent_error, "graph_recursion_limit: limit reached")
        self.assertEqual(result.missing_events[0].ticker, "AAPL")

    def test_eval_case_allows_unexpected_low_but_blocks_unexpected_medium(self) -> None:
        case = EvalCase(
            id="unit",
            portfolio_path="examples/golden_portfolio.json",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            expected_events=[
                ExpectedEvent(
                    ticker="AAPL",
                    event_type="downgrade",
                    catalyst_date=date(2024, 1, 2),
                    expected_materiality=Materiality.HIGH,
                )
            ],
        )

        def run_with_signals(signals: list[dict]) -> object:
            result = {
                "status": "ok",
                "rendered_brief": "brief",
                "signal_count": len(signals),
                "blocked_issues": [],
                "structured_brief": {"signals": signals},
            }
            with (
                patch("portfolio_threat_agent.evaluation.create_portfolio_threat_agent", return_value=object()),
                patch("portfolio_threat_agent.evaluation.run_portfolio_entries", return_value=([], [result])),
            ):
                return run_eval_case(case)

        expected = {
            "ticker": "AAPL",
            "event_type": "downgrade",
            "catalyst_date": "2024-01-02",
            "materiality": "HIGH",
        }
        low_result = run_with_signals([expected, {"ticker": "PFE", "event_type": "guidance", "catalyst_date": "2024-01-02", "materiality": "LOW"}])
        med_result = run_with_signals([expected, {"ticker": "PFE", "event_type": "guidance", "catalyst_date": "2024-01-02", "materiality": "MED"}])

        self.assertTrue(low_result.passed)
        self.assertFalse(med_result.passed)
        self.assertEqual(med_result.unexpected_high_medium_signals[0].ticker, "PFE")


if __name__ == "__main__":
    unittest.main()
