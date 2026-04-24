## 🌌 Guardrails Universe — Comprehensive Anthology of AI Safety Techniques

## 📌 Project Overview

**Guardrails Universe** is a curated collection of **AI safety, control, and reliability techniques** implemented using LangChain, LangGraph, and Azure OpenAI.  
Rather than focusing on a single application, this repository serves as a **technical anthology** — a structured, modular showcase of how different **guardrails** can be engineered, combined, and composed to shape LLM behavior.

Each guardrail is implemented as an independent, runnable module, allowing you to study, test, and understand each safety mechanism in isolation.  
A final integrated agent demonstrates how these guardrails can coexist harmoniously within a single system.

### Key Features
- **PII Guardrails** (redaction + masking)  
- **Content Filtering Guardrails**  
- **HITL (Human‑in‑the‑Loop) Action Safety**  
- **Context Summarization Guardrail**  
- **Preference Memory Guardrail**  
- **Structured Output Guardrail (Pydantic)**  
- **Reliability Middleware** (logging, retry, dynamic prompt)  
- **Integrated Agent combining all guardrails**  
- **Interactive Jupyter Notebook demo**

### Why This Matters
Modern AI systems require **layered safety**, not a single filter.  
Enterprises need agents that are:

- **safe** — prevent harmful or unintended actions  
- **predictable** — avoid hallucinations and inconsistent formats  
- **auditable** — provide traceability, human oversight and intervention  
- **composable** — support multiple guardrails without interference  
- **reliable** — recover gracefully from failures and behave predictably  

This repository demonstrates how to build such systems using **modular guardrails**, each addressing a different dimension of AI safety:

- Input safety  
- Output safety  
- Action safety  
- Context safety  
- Behavioral consistency  
- Engineering reliability  

The result is a practical, real‑world reference for designing **trustworthy AI agents**.

---

## 🛠️ Tech Stack

- **LangChain** — tools, agents, middleware  
- **LangGraph** — stateful execution, HITL, checkpointing  
- **Azure OpenAI** — model provider for all LLM calls  

---

## 📁 Repository Structure

```
src/
    agent_guardrails.py                   # integrated agent with all guardrails
    guardrail_content_filter.py           # content filtering guardrail
    guardrail_context_summarization.py    # context summarization guardrail
    guardrail_hitl.py                     # HITL (Human-in-the-Loop) action safety guardrail
    guardrail_memory_preference.py        # preference memory guardrail
    guardrail_pii.py                      # PII redaction/masking guardrail
    guardrail_structured_output.py        # structured output (Pydantic) guardrail
    guardrails_walkthrough.ipynb          # notebook demo with explanations
    knowledge_base.py                     # mock KB used by search_docs tool
    middleware_reliability.py             # logging, retry, dynamic prompt middleware
    tool.py                               # tool definitions (weather, calculator, email, KB search, user id)
.env.example                              # environment variable template
.gitignore                                # git ignore rules
README.md                                 # project documentation
requirements.txt                          # Python dependencies
```

---

## 🧭 Guardrails Overview

### Input Safety
- **PII Guardrails** — redact or mask sensitive data  
- **Content Filter Guardrails** — block unsafe or disallowed topics  

### Action Safety
- **HITL Guardrail** — pause execution and require human approval before executing sensitive tools (e.g., sending email)

### Output Safety
- **Structured Output Guardrail** — enforce Pydantic schema to prevent hallucinated formats

### Context Safety
- **Summarization Middleware** — compress long histories to avoid context overflow

### Behavior Consistency
- **Preference Memory Guardrail** — persist user preferences across turns

### Reliability
- **Logging Middleware** — before/after model hooks  
- **Retry Middleware** — auto‑retry transient failures  
- **Dynamic Prompt Middleware** — adjust system prompt based on conversation length  

---

## 🤖 Integrated Agent

`agent_guardrails.py` integrates:

- All guardrails  
- All reliability middleware  
- All tools  
- Checkpointing  
- User context  
- HITL resume flow  

This file demonstrates how multiple safety layers can coexist **without conflict**, forming a production‑grade agent architecture.

---

## 🚀 How to Run

### Install dependencies
```
pip install -r requirements.txt
```

### Set environment variables
Copy `.env.example` → `.env`  
Fill in Azure OpenAI credentials.

### Run individual guardrail demos
```
python src/guardrail_pii.py
python src/guardrail_hitl.py
python src/guardrail_structured_output.py
...
```

### Run the integrated agent
```
python src/agent_guardrails.py
```

### Open the notebook
```
src/guardrails_walkthrough.ipynb
```

---

## 🧪 Example Workflows

- **PII Redaction** — email → `[REDACTED]`  
- **Content Filter** — blocked keywords return safe fallback  
- **HITL Approval** — agent pauses before executing sensitive actions  
- **Structured Output** — enforced JSON schema  
- **Preference Memory** — agent recalls user style  
- **Summarization** — long histories compressed automatically  

---

## 🔮 Future Extensions

- Jailbreak detection  
- Semantic content moderation  
- RAG grounding guardrails  
- Tool sandboxing  
- Multi‑agent safety  
- Persistent preference memory (Redis/Postgres)  
