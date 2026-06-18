from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Any

from langsmith import traceable

from portfolio_threat_agent.structured import Source


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

URL_DATE_RE = re.compile(r"/(?P<year>20\d{2})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})(?:/|-)")
TEXT_DATE_RE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"(?P<day>\d{1,2}),\s+(?P<year>20\d{2})\b",
    re.IGNORECASE,
)


def _source_id(url: str) -> str:
    return "S" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def parse_source_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def infer_source_date(url: str, *texts: str | None) -> date | None:
    url_match = URL_DATE_RE.search(url)
    if url_match:
        try:
            return date(int(url_match.group("year")), int(url_match.group("month")), int(url_match.group("day")))
        except ValueError:
            pass

    combined = "\n".join(text for text in texts if text)
    for match in TEXT_DATE_RE.finditer(combined):
        try:
            return date(int(match.group("year")), MONTHS[match.group("month").lower().rstrip(".")], int(match.group("day")))
        except ValueError:
            continue
    return None


def _tool_name(message: Any) -> str:
    return str(getattr(message, "name", "") or "").lower()


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


@traceable(run_type="parser", name="extract_tavily_sources")
def extract_tavily_sources(messages: list[Any]) -> list[Source]:
    by_url: dict[str, Source] = {}
    for message in messages:
        if getattr(message, "type", None) != "tool" or "tavily" not in _tool_name(message):
            continue
        try:
            payload = json.loads(_message_text(message))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        query = str(payload.get("query", ""))
        for result in payload.get("results", []):
            if not isinstance(result, dict) or not result.get("url"):
                continue
            url = str(result["url"])
            published_date = parse_source_date(result.get("published_date") or result.get("date"))
            published_date = published_date or infer_source_date(
                url,
                result.get("title"),
                result.get("content"),
                result.get("raw_content"),
            )
            source = Source(
                source_id=_source_id(url),
                title=result.get("title") or "Untitled",
                url=url,
                query=query,
                content=result.get("content") or "",
                raw_content=result.get("raw_content"),
                published_date=published_date,
                score=result.get("score"),
            )
            existing = by_url.get(url)
            if existing is None or (source.raw_content and not existing.raw_content):
                by_url[url] = source
    return list(by_url.values())
