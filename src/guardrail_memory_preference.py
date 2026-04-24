import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

from langgraph.store.memory import InMemoryStore

from src.tool import get_weather, calculate, search_docs, Context

load_dotenv()


# -----------------------------
#   Preference Memory Guardrail
# -----------------------------
store = InMemoryStore()


@tool
def save_preference(style: str, runtime: ToolRuntime[Context]) -> str:
    """
    Save the user's preferred response style.
    This acts as a behavior-consistency guardrail:
    the agent will not forget or override user preferences.
    """
    runtime.store.put(("preferences",), runtime.context.user_id, {"style": style})
    return "Saved."


@tool
def read_preference(runtime: ToolRuntime[Context]) -> str:
    """
    Read the user's preferred response style.
    Ensures consistent behavior across turns.
    """
    pref = runtime.store.get(("preferences",), runtime.context.user_id)
    return pref.value.get("style", "balanced") if pref else "balanced"


def build_agent():
    """
    Build an agent with preference memory guardrails.
    The agent can store and recall user-specific settings,
    ensuring consistent behavior across conversations.
    """

    model = init_chat_model(
        model="gpt-4.1-mini",
        model_provider="azure_openai",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version="2025-01-01-preview",
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_docs, save_preference, read_preference],
        system_prompt="You are a helpful assistant. Use tools when needed.",
        store=store,
    )

    return agent


def demo():
    agent = build_agent()
    user = Context(user_id="raj713335")

    print("\n=== SAVE USER PREFERENCE ===")
    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My style is: super concise."
                }
            ]
        },
        context=user,
    )

    print("\n=== READ USER PREFERENCE ===")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What style do I prefer?"
                }
            ]
        },
        context=user,
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    demo()
