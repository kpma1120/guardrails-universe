from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from knowledge_base import search_kb


@dataclass
class Context:
    user_id: str


@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""

    # Replace with real API later
    return f"It's always sunny in {city}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '12 * (3 + 5)' """

    return str(eval(expression))


@tool
def search_docs(query: str) -> str:
    """Search internal support documentation for the given query."""
    return search_kb(query)


@tool
def get_user_id(runtime: ToolRuntime[Context]) -> str:
    """Return the current user_id from runtime context."""
    return runtime.context.user_id


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. (Demo tool: does not really send it.)"""
    return f"[SIMULATED] Email queued to {to} with subject '{subject}'"
