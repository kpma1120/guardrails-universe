import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from src.tool import get_weather, calculate, search_docs, Context

load_dotenv()


# -----------------------------
#   Structured Output Schema
# -----------------------------
class SupportActionPlan(BaseModel):
    """
    A structured plan for resolving a support request.
    This schema acts as an Output Safety Guardrail:
    - Prevents hallucinated formats
    - Ensures required fields exist
    - Forces predictable, machine-parseable output
    """

    summary: str = Field(description="1-2 sentence summary of the issue")
    steps: list[str] = Field(description="Concrete steps the user should take")
    needs_human: bool = Field(description="True if a human should review before action")


# -----------------------------
#   Build Agent
# -----------------------------
def build_agent():
    """
    Build an agent with a structured-output guardrail.
    The agent is forced to output JSON that matches SupportActionPlan.
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
        system_prompt="You are a helpful support assistant. Use tools when needed.",
        response_format=SupportActionPlan,   # <-- Structured Output Guardrail
    )

    return agent


# -----------------------------
#   Demo
# -----------------------------
def demo():
    agent = build_agent()
    user = Context(user_id="raj713335")

    print("\n=== STRUCTURED OUTPUT GUARDRAIL DEMO ===")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I can't reset my password. What do I do?"
                }
            ]
        },
        context=user,
    )

    print("\nStructured Response Object:")
    print(result["structured_response"])

    print("\nRaw Model Output:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    demo()
