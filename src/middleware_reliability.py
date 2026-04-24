import os
from dotenv import load_dotenv

from typing import Callable

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    before_model,
    after_model,
    wrap_model_call,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
)

from langgraph.runtime import Runtime

from src.tool import get_weather, calculate, search_docs

load_dotenv()


# -----------------------------
#   Logging Middleware
# -----------------------------
@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """
    Log the last user or system message before the model runs.
    Useful for debugging and observability.
    """
    last = state["messages"][-1]
    print(f"[before_model] last message: {getattr(last, 'content', last)}")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime):
    """
    Log the model's output message.
    Helps track model behavior and detect anomalies.
    """
    last = state["messages"][-1]
    print(f"[after_model] model output: {getattr(last, 'content', last)}")
    return None


# -----------------------------
#   Retry Middleware
# -----------------------------
@wrap_model_call
def retry_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """
    Retry model calls up to 3 times if transient errors occur.
    Improves reliability in production environments.
    """
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as ex:
            print(f"[retry] attempt {attempt + 1}/3 failed: {ex}")
            if attempt == 2:
                raise


# -----------------------------
#   Dynamic Prompt Middleware
# -----------------------------
@dynamic_prompt
def system_prompt_from_context(request: ModelRequest) -> str:
    """
    Adjust system prompt dynamically based on conversation length.
    Demonstrates adaptive prompt engineering.
    """
    if len(request.messages) > 10:
        return "You are a helpful assistant. Be extremely concise."
    return "You are a helpful assistant."


# -----------------------------
#   Build Agent
# -----------------------------
def build_agent():
    """
    Build an agent showcasing engineering reliability middleware:
    - Logging
    - Retry
    - Dynamic prompt
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
            log_before_model,
            log_after_model,
            system_prompt_from_context,
            retry_model,
        ],
    )

    return agent


# -----------------------------
#   Demo
# -----------------------------
def demo():
    agent = build_agent()

    print("\n=== RELIABILITY MIDDLEWARE DEMO ===")
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

    print("\nFinal Answer:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    demo()
