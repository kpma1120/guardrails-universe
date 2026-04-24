import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from src.tool import get_weather, calculate, search_docs

load_dotenv()


def build_agent():
    """
    Build an agent with a context-summarization guardrail.
    This guardrail prevents context-window overflow by summarizing
    older messages once the token count exceeds a threshold.
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
            SummarizationMiddleware(
                model="openai:gpt-4.1-mini",
                trigger=("tokens", 4000),   # summarize when context grows too large
                keep=("messages", 20),      # keep last 20 messages, summarize older ones
            )
        ],
    )

    return agent


def demo():
    agent = build_agent()

    print("\n=== CONTEXT SUMMARIZATION GUARDRAIL DEMO ===")
    print("Generating long conversation to trigger summarization...\n")

    # Simulate a long conversation
    messages = []
    for i in range(30):
        messages.append(
            {"role": "user", "content": f"Message number {i}. Please acknowledge."}
        )
        result = agent.invoke({"messages": messages})
        print(f"Assistant: {result['messages'][-1].content}")

    print("\n=== FINAL CONTEXT STATE ===")
    print("Older messages should now be summarized to prevent context overflow.")


if __name__ == "__main__":
    demo()
