import os
from dataclasses import dataclass
from dotenv import load_dotenv

from typing import Callable, Any
from pydantic import BaseModel, Field

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
    PIIMiddleware,
    AgentMiddleware,
    hook_config,
    HumanInTheLoopMiddleware,
    SummarizationMiddleware,
)

from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from langchain.tools import tool, ToolRuntime

from src.tool import (
    get_weather,
    calculate,
    search_docs,
    get_user_id,
    send_email,
    Context,
)

load_dotenv()

# ============================================================
# 1. Context Dataclass
# ============================================================

@dataclass
class Context:
    user_id: str


# ============================================================
# 2. Memory Store (for preference guardrail)
# ============================================================

store = InMemoryStore()


# ============================================================
# 3. Content Filter Guardrail
# ============================================================

class ContentFilterMiddleware(AgentMiddleware):
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


# ============================================================
# 4. Preference Memory Guardrail
# ============================================================

@tool
def save_preference(style: str, runtime: ToolRuntime[Context]) -> str:
    store.put(("preferences",), runtime.context.user_id, {"style": style})
    return "Saved."


@tool
def read_preference(runtime: ToolRuntime[Context]) -> str:
    pref = runtime.store.get(("preferences",), runtime.context.user_id)
    return pref.value.get("style", "balanced") if pref else "balanced"


# ============================================================
# 5. Structured Output Guardrail (optional)
# ============================================================

class SupportActionPlan(BaseModel):
    summary: str = Field(description="1-2 sentence summary of the issue")
    steps: list[str] = Field(description="Concrete steps the user should take")
    needs_human: bool = Field(description="True if a human should review before action")


# ============================================================
# 6. Reliability Middleware (logging + retry + dynamic prompt)
# ============================================================

@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    last = state["messages"][-1]
    print(f"[before_model] last message: {getattr(last, 'content', last)}")


@after_model
def log_after_model(state: AgentState, runtime: Runtime):
    last = state["messages"][-1]
    print(f"[after_model] model output: {getattr(last, 'content', last)}")


@wrap_model_call
def retry_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as ex:
            print(f"[retry] attempt {attempt+1}/3 failed: {ex}")
            if attempt == 2:
                raise


@dynamic_prompt
def system_prompt_from_context(request: ModelRequest) -> str:
    if len(request.messages) > 10:
        return "You are a helpful assistant. Be extremely concise."
    return "You are a helpful assistant."


# ============================================================
# 7. Build Full-Stack Agent
# ============================================================

def build_agent():
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
        tools=[
            get_weather,
            calculate,
            search_docs,
            get_user_id,
            save_preference,
            read_preference,
            send_email,
        ],
        system_prompt="You are a helpful support assistant. Use tools when needed.",
        checkpointer=checkpointer,
        store=store,
        # response_format=SupportActionPlan,  # optional structured output guardrail
        middleware=[
            # --- Safety Guardrails ---
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            ContentFilterMiddleware(["hotel"]),
            HumanInTheLoopMiddleware(
                interrupt_on={"send_email": True, "search_docs": False}
            ),

            # --- Context Safety ---
            SummarizationMiddleware(
                model="openai:gpt-4.1-mini",
                trigger=("tokens", 4000),
                keep=("messages", 20),
            ),

            # --- Reliability Middleware ---
            log_before_model,
            log_after_model,
            system_prompt_from_context,
            retry_model,
        ],
    )

    return agent


# ============================================================
# 8. Demo
# ============================================================

def demo():
    agent = build_agent()
    user = Context(user_id="raj713335")

    print("\n=== FULL STACK AGENT DEMO ===")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Email the team: subject 'Update', body 'All Good Folks!'"
                }
            ]
        },
        context=user,
        config={"configurable": {"thread_id": "demo"}},
    )

    print("\nINTERRUPT:", result.get("__interrupt__"))

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config={"configurable": {"thread_id": "demo"}},
    )

    print("\nFINAL ANSWER:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    demo()
