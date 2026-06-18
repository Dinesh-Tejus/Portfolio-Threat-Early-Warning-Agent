from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Annotated

from rich.console import Console
from rich.panel import Panel
import typer

from portfolio_threat_agent.agent import ask_agent, monitor_portfolio
from portfolio_threat_agent.config import create_monitor_config_store, handle_monitor_command
from portfolio_threat_agent.evaluation import run_eval_dataset
from portfolio_threat_agent.models import has_openai_api_key
from portfolio_threat_agent.monitor import PortfolioThreatMonitor
from portfolio_threat_agent.telegram import run_telegram_polling


app = typer.Typer(add_completion=False, help="Local runner for the portfolio threat agent.")
console = Console()


def _stream_event(text: str) -> None:
    console.print(f"[dim cyan]{text}[/dim cyan]")


def _require_env(name: str, help_text: str) -> None:
    if os.getenv(name):
        return
    console.print(f"[bold red]Missing {name}[/bold red]")
    console.print(help_text)
    raise typer.Exit(code=1)


def _require_openai_api_key() -> None:
    if has_openai_api_key():
        return
    console.print("[bold red]Missing OPENAI_API_KEY[/bold red]")
    console.print("Set OPENAI_API_KEY to your OpenAI API key.")
    raise typer.Exit(code=1)


@app.command()
def ask(
    portfolio: Annotated[Path, typer.Option("--portfolio", "-p", exists=True, readable=True, help="Path to portfolio.json.")],
    question: Annotated[list[str], typer.Argument(help="Question or monitoring instruction.")],
    since: Annotated[str, typer.Option(help="Monitoring window, e.g. 7d.")] = "7d",
    virtual_date: Annotated[str | None, typer.Option(help="Compatibility active date YYYY-MM-DD; prefer --start-date/--end-date.")] = None,
    start_date: Annotated[str | None, typer.Option(help="Historical retrieval start date YYYY-MM-DD.")] = None,
    end_date: Annotated[str | None, typer.Option(help="Historical active/retrieval end date YYYY-MM-DD.")] = None,
    max_results: Annotated[int, typer.Option(help="Max Tavily results per search.")] = 5,
) -> None:
    """Ask the portfolio threat agent. Uses the same structured citation contract as monitor."""
    _require_env("TAVILY_API_KEY", "Create one at https://app.tavily.com, then set TAVILY_API_KEY.")
    _require_openai_api_key()

    question_text = " ".join(question)
    console.print(Panel.fit(question_text, title="Agent Input", border_style="cyan"))
    active_date = date.fromisoformat(virtual_date) if virtual_date else None
    answer = ask_agent(
        question_text,
        portfolio_path=portfolio,
        since=since,
        max_results=max_results,
        virtual_date=active_date,
        start_date=date.fromisoformat(start_date) if start_date else None,
        end_date=date.fromisoformat(end_date) if end_date else None,
        stream_handler=_stream_event,
    )
    console.print(Panel(answer or "No answer returned.", title="Agent", border_style="green"))


@app.command()
def monitor(
    portfolio: Annotated[Path, typer.Option("--portfolio", "-p", exists=True, readable=True, help="Path to portfolio.json.")],
    instruction: Annotated[list[str] | None, typer.Argument(help="Optional extra instruction.")] = None,
    since: Annotated[str, typer.Option(help="Monitoring window, e.g. 7d.")] = "7d",
    virtual_date: Annotated[str | None, typer.Option(help="Compatibility active date YYYY-MM-DD; prefer --start-date/--end-date.")] = None,
    start_date: Annotated[str | None, typer.Option(help="Historical retrieval start date YYYY-MM-DD.")] = None,
    end_date: Annotated[str | None, typer.Option(help="Historical active/retrieval end date YYYY-MM-DD.")] = None,
    max_results: Annotated[int, typer.Option(help="Max Tavily results per search.")] = 5,
) -> None:
    """Give the agent a portfolio.json. The agent decides ticker/sector searches."""
    _require_env("TAVILY_API_KEY", "Create one at https://app.tavily.com, then set TAVILY_API_KEY.")
    _require_openai_api_key()

    extra = " ".join(instruction) if instruction else None
    console.print(Panel.fit(str(portfolio), title="Portfolio", border_style="cyan"))
    active_date = date.fromisoformat(virtual_date) if virtual_date else None
    answer = monitor_portfolio(
        portfolio,
        since=since,
        instruction=extra,
        max_results=max_results,
        virtual_date=active_date,
        start_date=date.fromisoformat(start_date) if start_date else None,
        end_date=date.fromisoformat(end_date) if end_date else None,
        stream_handler=_stream_event,
    )
    console.print(Panel(answer or "No answer returned.", title="Agent", border_style="green"))


@app.command()
def eval(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", exists=True, readable=True, help="Path to eval dataset JSON.")],
    max_results: Annotated[int, typer.Option(help="Max Tavily results per search.")] = 5,
) -> None:
    """Run historical agent-evaluation cases."""
    _require_env("TAVILY_API_KEY", "Create one at https://app.tavily.com, then set TAVILY_API_KEY.")
    _require_openai_api_key()

    report = run_eval_dataset(dataset, max_results=max_results)
    console.print_json(data=report.model_dump(mode="json"))


@app.command()
def config(
    text: Annotated[list[str], typer.Argument(help="Command, e.g. '/change_interval 30 mins'.")],
) -> None:
    """Simulate a monitor command."""
    response = handle_monitor_command(" ".join(text))
    console.print(response)


@app.command()
def run_monitor(
    portfolio: Annotated[Path, typer.Option("--portfolio", "-p", exists=True, readable=True, help="Path to portfolio.json.")],
    once: Annotated[bool, typer.Option(help="Run one monitor tick and exit.")] = False,
) -> None:
    """Run the long-lived monitor. The agent is created once and config is reread between ticks."""
    store = create_monitor_config_store()
    current = store.load().model_copy(update={"portfolio_path": str(portfolio)})
    store.save(current)
    monitor = PortfolioThreatMonitor(store)
    if once:
        console.print(Panel(monitor.run_once(), title="Agent", border_style="green"))
        return
    monitor.run_forever()


@app.command()
def telegram_bot() -> None:
    """Run Telegram long-polling ingress for commands and portfolio JSON messages."""
    _require_env("TELEGRAM_BOT_TOKEN", "Create a bot with BotFather, then set TELEGRAM_BOT_TOKEN.")
    monitor = PortfolioThreatMonitor()
    run_telegram_polling(monitor)


if __name__ == "__main__":
    app()
