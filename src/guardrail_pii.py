import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

from src.tool import get_weather, calculate, search_docs, Context

load_dotenv()


def build_agent():
    """
    Build an agent with PII guardrails:
    - Email redaction
    - Credit card masking
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
            # --- PII Guardrails ---
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        ],
    )

    return agent


def demo():
    agent = build_agent()

    print("\n=== PII REDACTION DEMO ===")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My email is john.doe@example.com. What is the weather in Delhi?"
                }
            ]
        }
    )
    print(result["messages"][-1].content)

    print("\n=== CREDIT CARD MASKING DEMO ===")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My credit card number is 4242 4242 4242 4242. What is 3+5?"
                }
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    demo()
