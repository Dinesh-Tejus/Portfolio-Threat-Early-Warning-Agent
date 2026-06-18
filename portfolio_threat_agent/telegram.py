from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib import request

from langsmith import traceable

from portfolio_threat_agent.config import MonitorConfigStore, create_monitor_config_store, handle_monitor_command
from portfolio_threat_agent.portfolio import Portfolio


TELEGRAM_API_BASE = "https://api.telegram.org"
COMMANDS = {"/change_interval", "/set_monitoring", "/check_now", "/status"}


def _bot_token() -> str | None:
    return os.getenv("TELEGRAM_BOT_TOKEN")


def _default_chat_id() -> str | None:
    return os.getenv("TELEGRAM_CHAT_ID") or create_monitor_config_store().load().telegram_chat_id


def telegram_api(method: str, payload: dict[str, object], token: str | None = None, socket_timeout: int = 10) -> dict[str, object]:
    bot_token = token or _bot_token()
    if not bot_token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN is not configured"}

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{TELEGRAM_API_BASE}/bot{bot_token}/{method}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=socket_timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@traceable(run_type="tool", name="send_telegram_message")
def send_telegram_message(text: str, chat_id: str | None = None) -> dict[str, object]:
    """Post a plain text message to Telegram with the bot token."""
    if not _bot_token():
        return {"status": "skipped", "reason": "TELEGRAM_BOT_TOKEN is not configured"}

    target_chat = chat_id or _default_chat_id()
    if not target_chat:
        return {"status": "skipped", "reason": "No Telegram chat configured"}

    response = telegram_api("sendMessage", {"chat_id": target_chat, "text": text})
    if not response.get("ok"):
        return {"status": "error", "response": response}
    result = response.get("result", {})
    message_id = result.get("message_id") if isinstance(result, dict) else None
    return {"status": "sent", "chat_id": str(target_chat), "message_id": message_id}


@traceable(run_type="chain", name="handle_telegram_config_text")
def handle_telegram_config_text(text: str) -> str:
    """Handle Telegram text that changes runtime monitor configuration."""
    return handle_monitor_command(text)


def _portfolio_path(chat_id: str, user_id: str, root: Path = Path("state/portfolios")) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{chat_id}_{user_id}")
    return root / f"{safe}.json"


@traceable(run_type="chain", name="save_telegram_portfolio_json")
def save_telegram_portfolio_json(
    text: str,
    chat_id: str,
    user_id: str,
    store: MonitorConfigStore,
    root: Path = Path("state/portfolios"),
) -> str:
    payload = json.loads(text)
    portfolio = Portfolio.model_validate(payload)
    path = _portfolio_path(chat_id, user_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(portfolio.model_dump_json(indent=2))
    store.set_portfolio_path(str(path))
    store.set_telegram_chat(chat_id)
    return f"Portfolio saved with {len(portfolio.positions)} positions."


def dispatch_telegram_command(text: str, chat_id: str, monitor: "PortfolioThreatMonitor") -> str:
    command, _, args = text.strip().partition(" ")
    command = command.split("@", 1)[0].lower()
    if command not in COMMANDS:
        return "Unknown command. Use /change_interval, /set_monitoring, /check_now, or /status."
    monitor.config_store.set_telegram_chat(chat_id)
    return monitor.handle_command(f"{command} {args}".strip(), chat_id=chat_id)


def _message_from_update(update: dict[str, Any]) -> dict[str, Any] | None:
    message = update.get("message") or update.get("edited_message")
    return message if isinstance(message, dict) else None


def _chat_id(message: dict[str, Any]) -> str | None:
    chat = message.get("chat")
    if not isinstance(chat, dict) or chat.get("id") is None:
        return None
    return str(chat["id"])


def _user_id(message: dict[str, Any]) -> str:
    user = message.get("from")
    if isinstance(user, dict) and user.get("id") is not None:
        return str(user["id"])
    return "unknown-user"


@traceable(run_type="chain", name="handle_telegram_message")
def handle_telegram_message(text: str, chat_id: str, user_id: str, monitor: "PortfolioThreatMonitor") -> str:
    stripped = text.strip()
    if not stripped:
        return "Send a command or portfolio JSON."
    if stripped.startswith("/"):
        return dispatch_telegram_command(stripped, chat_id, monitor)
    if stripped.startswith("{"):
        try:
            return save_telegram_portfolio_json(stripped, chat_id, user_id, monitor.config_store)
        except Exception as exc:
            return f"Could not save portfolio JSON: {exc}"
    return "Send portfolio JSON or use /change_interval, /set_monitoring, /check_now, or /status."


def poll_telegram_once(monitor: "PortfolioThreatMonitor", offset: int | None = None, timeout: int = 30) -> int | None:
    payload: dict[str, object] = {"timeout": timeout, "allowed_updates": ["message", "edited_message"]}
    if offset is not None:
        payload["offset"] = offset
    response = telegram_api("getUpdates", payload, socket_timeout=timeout + 10)
    if not response.get("ok"):
        return offset

    next_offset = offset
    for update in response.get("result", []):
        if not isinstance(update, dict):
            continue
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            next_offset = update_id + 1
        message = _message_from_update(update)
        if not message:
            continue
        chat_id = _chat_id(message)
        text = message.get("text")
        if chat_id is None or not isinstance(text, str):
            continue
        reply = handle_telegram_message(text, chat_id, _user_id(message), monitor)
        send_telegram_message(reply, chat_id=chat_id)
    return next_offset


def run_telegram_polling(monitor: "PortfolioThreatMonitor") -> None:
    offset: int | None = None
    while True:
        try:
            offset = poll_telegram_once(monitor, offset=offset)
        except Exception as exc:
            print(f"[telegram] polling error ({type(exc).__name__}): {exc} - retrying in 5s")
            time.sleep(5)
        else:
            time.sleep(1)
