import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.tool import (
    get_weather,
    calculate,
    search_docs,
    get_user_id,
    send_email,
    Context,
)

load_dotenv()


def build_agent():
    """
    Build an agent with Human-in-the-Loop (HITL) guardrails.
    HITL prevents the agent from automatically executing sensitive tools
    such as send_email, requiring human approval before continuing.
    """

    model = init_chat_model(
        model="gpt-4.1-mini",
        model_provider="azure_openai",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version="2025-01-01-preview",
    )

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_docs, get_user_id, send_email],
        system_prompt="You are a helpful assistant. Use tools when needed.",
        checkpointer=checkpointer,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": True,   # require approval
                    "search_docs": False  # allow auto-execution
                }
            )
        ],
    )

    return agent


def demo():
    agent = build_agent()

    print("\n=== HITL INTERRUPT DEMO ===")
    config = {"configurable": {"thread_id": "hitl-demo"}}

    # Step 1 — user triggers a sensitive tool (send_email)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Email the team: subject 'Update', body 'All Good Folks!'"
                }
            ]
        },
        context=Context(user_id="raj713335"),
        config=config,
    )

    print("INTERRUPT:", result.get("__interrupt__"))

    # Step 2 — human approves the action
    print("\n=== HITL APPROVE ===")
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
    )
    print(result["messages"][-1].content)

    # Step 3 — human rejects the action
    print("\n=== HITL REJECT ===")
    config = {"configurable": {"thread_id": "hitl-reject"}}

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Email the team: subject 'Update', body 'All Good Folks!'"
                }
            ]
        },
        context=Context(user_id="raj713335"),
        config=config,
    )

    print("INTERRUPT:", result.get("__interrupt__"))

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "reject"}]}),
        config=config,
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    demo()
