import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config

from langgraph.runtime import Runtime

from src.tool import get_weather, calculate, search_docs, Context

load_dotenv()


# -----------------------------
#   Content Filtering Guardrail
# -----------------------------
class ContentFilterMiddleware(AgentMiddleware):
    """
    A simple rule-based content filter.
    If the user message contains any banned keyword,
    the agent immediately stops and returns a safe response.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned = [b.lower() for b in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        if not state["messages"]:
            return None

        content = getattr(state["messages"][0], "content", "").lower()

        if any(b in content for b in self.banned):
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I can't help with that request. Please rephrase."
                    }
                ],
                "jump_to": "end"
            }

        return None


def build_agent():
    """
    Build an agent with a deterministic content filtering guardrail.
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
        tools=[get_weather, calculate, search_docs],
        system_prompt="You are a helpful assistant. Use tools when needed.",
        middleware=[
            ContentFilterMiddleware(["hotel", "gambling", "hack"]),
        ],
    )

    return agent


def demo():
    agent = build_agent()

    print("\n=== SAFE QUERY (ALLOWED) ===")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Delhi?"
                }
            ]
        }
    )
    print(result["messages"][-1].content)

    print("\n=== BLOCKED QUERY (BANNED KEYWORD: 'hotel') ===")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Find me the best hotel in my area."
                }
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    demo()
