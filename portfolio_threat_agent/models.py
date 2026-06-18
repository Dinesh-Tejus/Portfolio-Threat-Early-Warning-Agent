from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


DEFAULT_MODEL = "gpt-4o-mini"


def has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def create_chat_model(model: str = DEFAULT_MODEL, temperature: float = 0):
    return ChatOpenAI(model=model, temperature=temperature)
