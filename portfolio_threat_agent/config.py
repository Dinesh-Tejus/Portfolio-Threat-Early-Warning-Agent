from __future__ import annotations

import json
import os
from pathlib import Path
import re
from collections.abc import Callable
from typing import Any

from langsmith import traceable
from pydantic import BaseModel, Field, field_validator

from portfolio_threat_agent.models import DEFAULT_MODEL


DEFAULT_CONFIG_PATH = Path("state/monitor_config.json")
DEFAULT_STATE_PATH = Path("state/monitor_state.json")
DEFAULT_INTERVAL_SECONDS = 15 * 60
DEFAULT_MONGO_DATABASE = "portfolio_threat_agent"
DEFAULT_MONGO_COLLECTION = "monitor"
DEFAULT_MONGO_CONFIG_ID = "config:default"
DEFAULT_MONGO_STATE_ID = "state:default"

INTERVAL_RE = re.compile(
    r"(?P<count>\d+)\s*(?P<unit>s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)",
    re.IGNORECASE,
)


class MonitorConfig(BaseModel):
    monitoring_enabled: bool = True
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS
    since: str = "7d"
    instruction: str | None = None
    portfolio_path: str | None = None
    telegram_chat_id: str | None = None
    model: str = DEFAULT_MODEL
    max_results: int = 5

    @field_validator("model", mode="before")
    @classmethod
    def force_default_model(cls, value: object) -> str:
        return DEFAULT_MODEL


class MonitorState(BaseModel):
    alerted_source_urls: list[str] = Field(default_factory=list)


def _mongo_document_id(kind: str, explicit: str | None, default: str) -> str:
    if explicit:
        return explicit
    monitor_id = os.getenv("MONGODB_MONITOR_ID")
    if monitor_id:
        return f"{kind}:{monitor_id}"
    return default


@traceable(run_type="parser", name="parse_interval_seconds")
def parse_interval_seconds(text: str) -> int:
    match = INTERVAL_RE.search(text)
    if not match:
        raise ValueError("Could not find an interval like 30m, 30 minutes, or 1h.")

    count = int(match.group("count"))
    unit = match.group("unit").lower()
    if unit.startswith("s"):
        seconds = count
    elif unit.startswith("m"):
        seconds = count * 60
    else:
        seconds = count * 60 * 60

    if seconds < 60:
        raise ValueError("Monitoring interval must be at least 60 seconds.")
    return seconds


def format_interval(seconds: int) -> str:
    if seconds % 3600 == 0:
        value = seconds // 3600
        return f"{value} hour" + ("" if value == 1 else "s")
    if seconds % 60 == 0:
        value = seconds // 60
        return f"{value} minute" + ("" if value == 1 else "s")
    return f"{seconds} seconds"


class MonitorConfigStore:
    def __init__(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.path = path

    @traceable(run_type="chain", name="load_monitor_config")
    def load(self) -> MonitorConfig:
        if not self.path.exists():
            return MonitorConfig()
        return MonitorConfig.model_validate(json.loads(self.path.read_text()))

    @traceable(run_type="chain", name="save_monitor_config")
    def save(self, config: MonitorConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(config.model_dump_json(indent=2))

    @traceable(run_type="chain", name="set_monitor_interval")
    def set_interval(self, interval_seconds: int) -> MonitorConfig:
        config = self.load().model_copy(update={"interval_seconds": interval_seconds})
        self.save(config)
        return config

    @traceable(run_type="chain", name="set_monitoring_enabled")
    def set_monitoring_enabled(self, enabled: bool) -> MonitorConfig:
        config = self.load().model_copy(update={"monitoring_enabled": enabled})
        self.save(config)
        return config

    @traceable(run_type="chain", name="set_portfolio_path")
    def set_portfolio_path(self, portfolio_path: str) -> MonitorConfig:
        config = self.load().model_copy(update={"portfolio_path": portfolio_path})
        self.save(config)
        return config

    @traceable(run_type="chain", name="set_telegram_chat")
    def set_telegram_chat(self, chat_id: str) -> MonitorConfig:
        config = self.load().model_copy(update={"telegram_chat_id": chat_id})
        self.save(config)
        return config


class MongoMonitorConfigStore:
    def __init__(
        self,
        uri: str | None = None,
        database: str | None = None,
        collection: str | None = None,
        document_id: str | None = None,
        collection_obj: Any | None = None,
    ) -> None:
        self.document_id = document_id or _mongo_document_id("config", os.getenv("MONGODB_MONITOR_CONFIG_ID"), DEFAULT_MONGO_CONFIG_ID)
        if collection_obj is not None:
            self.collection = collection_obj
            return

        mongo_uri = uri or os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MONGODB_URI is required for MongoMonitorConfigStore.")
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError("pymongo is required for Mongo persistence. Install project dependencies or `pip install pymongo`.") from exc

        client = MongoClient(mongo_uri)
        db_name = database or os.getenv("MONGODB_DATABASE") or DEFAULT_MONGO_DATABASE
        collection_name = collection or os.getenv("MONGODB_COLLECTION") or DEFAULT_MONGO_COLLECTION
        self.collection = client[db_name][collection_name]

    @traceable(run_type="chain", name="load_monitor_config_mongo")
    def load(self) -> MonitorConfig:
        document = self.collection.find_one({"_id": self.document_id})
        if not document:
            return MonitorConfig()
        payload = {key: value for key, value in document.items() if key != "_id"}
        return MonitorConfig.model_validate(payload)

    @traceable(run_type="chain", name="save_monitor_config_mongo")
    def save(self, config: MonitorConfig) -> None:
        self.collection.update_one(
            {"_id": self.document_id},
            {"$set": config.model_dump(mode="json")},
            upsert=True,
        )

    @traceable(run_type="chain", name="set_monitor_interval_mongo")
    def set_interval(self, interval_seconds: int) -> MonitorConfig:
        config = self.load().model_copy(update={"interval_seconds": interval_seconds})
        self.save(config)
        return config

    @traceable(run_type="chain", name="set_monitoring_enabled_mongo")
    def set_monitoring_enabled(self, enabled: bool) -> MonitorConfig:
        config = self.load().model_copy(update={"monitoring_enabled": enabled})
        self.save(config)
        return config

    @traceable(run_type="chain", name="set_portfolio_path_mongo")
    def set_portfolio_path(self, portfolio_path: str) -> MonitorConfig:
        config = self.load().model_copy(update={"portfolio_path": portfolio_path})
        self.save(config)
        return config

    @traceable(run_type="chain", name="set_telegram_chat_mongo")
    def set_telegram_chat(self, chat_id: str) -> MonitorConfig:
        config = self.load().model_copy(update={"telegram_chat_id": chat_id})
        self.save(config)
        return config


class MonitorStateStore:
    def __init__(self, path: Path = DEFAULT_STATE_PATH) -> None:
        self.path = path

    @traceable(run_type="chain", name="load_monitor_state")
    def load(self) -> MonitorState:
        if not self.path.exists():
            return MonitorState()
        return MonitorState.model_validate(json.loads(self.path.read_text()))

    @traceable(run_type="chain", name="save_monitor_state")
    def save(self, state: MonitorState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(state.model_dump_json(indent=2))

    @traceable(run_type="chain", name="remember_alerted_sources")
    def remember_source_urls(self, urls: set[str]) -> MonitorState:
        state = self.load()
        merged = sorted(set(state.alerted_source_urls) | urls)
        updated = state.model_copy(update={"alerted_source_urls": merged})
        self.save(updated)
        return updated


class MongoMonitorStateStore:
    def __init__(
        self,
        uri: str | None = None,
        database: str | None = None,
        collection: str | None = None,
        document_id: str | None = None,
        collection_obj: Any | None = None,
    ) -> None:
        self.document_id = document_id or _mongo_document_id("state", os.getenv("MONGODB_MONITOR_STATE_ID"), DEFAULT_MONGO_STATE_ID)
        if collection_obj is not None:
            self.collection = collection_obj
            return

        mongo_uri = uri or os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MONGODB_URI is required for MongoMonitorStateStore.")
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError("pymongo is required for Mongo persistence. Install project dependencies or `pip install pymongo`.") from exc

        client = MongoClient(mongo_uri)
        db_name = database or os.getenv("MONGODB_DATABASE") or DEFAULT_MONGO_DATABASE
        collection_name = collection or os.getenv("MONGODB_COLLECTION") or DEFAULT_MONGO_COLLECTION
        self.collection = client[db_name][collection_name]

    @traceable(run_type="chain", name="load_monitor_state_mongo")
    def load(self) -> MonitorState:
        document = self.collection.find_one({"_id": self.document_id})
        if not document:
            return MonitorState()
        payload = {key: value for key, value in document.items() if key != "_id"}
        return MonitorState.model_validate(payload)

    @traceable(run_type="chain", name="save_monitor_state_mongo")
    def save(self, state: MonitorState) -> None:
        self.collection.update_one(
            {"_id": self.document_id},
            {"$set": state.model_dump(mode="json")},
            upsert=True,
        )

    @traceable(run_type="chain", name="remember_alerted_sources_mongo")
    def remember_source_urls(self, urls: set[str]) -> MonitorState:
        state = self.load()
        merged = sorted(set(state.alerted_source_urls) | urls)
        updated = state.model_copy(update={"alerted_source_urls": merged})
        self.save(updated)
        return updated


def create_monitor_config_store() -> MonitorConfigStore | MongoMonitorConfigStore:
    if os.getenv("MONGODB_URI"):
        return MongoMonitorConfigStore()
    return MonitorConfigStore()


def create_monitor_state_store() -> MonitorStateStore | MongoMonitorStateStore:
    if os.getenv("MONGODB_URI"):
        return MongoMonitorStateStore()
    return MonitorStateStore()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "on", "yes", "1"}:
        return True
    if normalized in {"false", "off", "no", "0"}:
        return False
    raise ValueError("Expected True or False.")


@traceable(run_type="chain", name="handle_monitor_command")
def handle_monitor_command(
    text: str,
    store: MonitorConfigStore | None = None,
    check_now: Callable[[], str] | None = None,
) -> str:
    store = store or create_monitor_config_store()
    stripped = text.strip()
    if not stripped:
        return "Unknown command. Use /change_interval, /set_monitoring, or /check_now."

    command, _, args = stripped.partition(" ")
    command = command.lower()
    args = args.strip()

    if command == "/change_interval":
        if not args:
            return "Usage: /change_interval 30 mins"
        try:
            interval_seconds = parse_interval_seconds(args)
        except ValueError as exc:
            return f"Usage: /change_interval 30 mins ({exc})"
        config = store.set_interval(interval_seconds)
        return f"Monitoring interval updated to {format_interval(config.interval_seconds)}."

    if command == "/set_monitoring":
        if not args:
            return "Usage: /set_monitoring True or /set_monitoring False"
        try:
            enabled = parse_bool(args)
        except ValueError as exc:
            return f"Usage: /set_monitoring True or /set_monitoring False ({exc})"
        config = store.set_monitoring_enabled(enabled)
        state = "enabled" if config.monitoring_enabled else "disabled"
        return f"Monitoring {state}."

    if command == "/check_now":
        if check_now is None:
            return "Check requested, but no monitor runner is attached yet."
        return check_now()

    if command == "/status":
        config = store.load()
        state = "enabled" if config.monitoring_enabled else "disabled"
        return f"Monitoring {state}; interval {format_interval(config.interval_seconds)}."

    return "Unknown command. Use /change_interval, /set_monitoring, or /check_now."
