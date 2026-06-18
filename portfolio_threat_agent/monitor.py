from __future__ import annotations

from datetime import date
from pathlib import Path
import threading

from langsmith import traceable

from portfolio_threat_agent.agent import (
    combine_validated_results,
    configure_langsmith,
    create_portfolio_threat_agent,
    run_portfolio_entries,
    validated_signal_count,
    validated_source_urls,
    window_start_date,
)
from portfolio_threat_agent.portfolio import load_portfolio
from portfolio_threat_agent.telegram import send_telegram_message
from portfolio_threat_agent.config import MonitorConfigStore, MonitorStateStore, create_monitor_config_store, create_monitor_state_store


def _monitor_context(instruction: str | None) -> str:
    return instruction.strip() if instruction else ""


class PortfolioThreatMonitor:
    """Long-running monitor that keeps one agent instance alive."""

    def __init__(self, config_store: MonitorConfigStore | None = None, state_store: MonitorStateStore | None = None) -> None:
        configure_langsmith()
        self.config_store = config_store or create_monitor_config_store()
        self.state_store = state_store or create_monitor_state_store()
        config = self.config_store.load()
        self.agent = create_portfolio_threat_agent(model=config.model, max_results=config.max_results)
        self.model = config.model
        self.max_results = config.max_results
        self._stop_event = threading.Event()
        self._reset_event = threading.Event()
        self._tick_lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._pending_chat_id: str | None = None
        self._pending_lock = threading.Lock()
        if config.monitoring_enabled and config.portfolio_path:
            self.start_monitor()

    def start_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._stop_event.set()
        self._reset_event.set()

    def _sleep_until_next_tick(self) -> None:
        while not self._stop_event.is_set():
            interval = self.config_store.load().interval_seconds
            self._reset_event.clear()
            woken_by_reset = self._reset_event.wait(timeout=interval)
            if not woken_by_reset:
                return

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            rendered = self.run_once()
            with self._pending_lock:
                pending = self._pending_chat_id
                self._pending_chat_id = None
            if pending:
                send_telegram_message(rendered, chat_id=pending)
            self._sleep_until_next_tick()

    @traceable(run_type="chain", name="monitor_tick")
    def run_once(self, force: bool = False) -> str:
        if not self._tick_lock.acquire(blocking=not force):
            return "A check is already in progress."
        try:
            config = self.config_store.load()
            if not force and not config.monitoring_enabled:
                return "Monitoring is disabled."

            if not config.portfolio_path:
                return "No portfolio configured."

            if config.model != self.model or config.max_results != self.max_results:
                self.agent = create_portfolio_threat_agent(model=config.model, max_results=config.max_results)
                self.model = config.model
                self.max_results = config.max_results

            active_date = date.today()
            retrieval_start = window_start_date(config.since, active_date)
            portfolio = load_portfolio(Path(config.portfolio_path))
            _messages, results = run_portfolio_entries(
                self.agent,
                portfolio,
                since=config.since,
                instruction=_monitor_context(config.instruction),
                active_date=active_date,
                retrieval_start=retrieval_start,
            )
            rendered = combine_validated_results(results)
            has_signals = validated_signal_count(results) > 0
            if has_signals:
                source_urls = validated_source_urls(results)
                seen_urls = set(self.state_store.load().alerted_source_urls)
                new_urls = source_urls - seen_urls
                if not force and new_urls:
                    send_telegram_message(rendered)
                self.state_store.remember_source_urls(source_urls)
            return rendered
        finally:
            self._tick_lock.release()

    def run_forever(self) -> None:
        self.start_monitor()
        if self._monitor_thread:
            self._monitor_thread.join()

    def _check_now_and_reset(self, chat_id: str | None = None) -> str:
        result = self.run_once(force=True)
        if result == "A check is already in progress." and chat_id:
            with self._pending_lock:
                self._pending_chat_id = chat_id
        else:
            self._reset_event.set()
        return result

    def handle_command(self, text: str, chat_id: str | None = None) -> str:
        from portfolio_threat_agent.config import handle_monitor_command

        command = text.strip().partition(" ")[0].lower()

        if command == "/set_monitoring":
            result = handle_monitor_command(text, store=self.config_store)
            if self.config_store.load().monitoring_enabled:
                self.start_monitor()
            else:
                self.stop_monitor()
            return result

        return handle_monitor_command(
            text,
            store=self.config_store,
            check_now=lambda: self._check_now_and_reset(chat_id=chat_id),
        )
