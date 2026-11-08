# Self-Evaluating RAG Content Engine (Local/Free Edition)

**Assessment Deliverable:** GenAI Engineer – Content Systems Take-Home Assessment
**Target Learner:** 12th Grade Pass Student from India (Non-English Background, No Prior Coding Experience)
**Core Paradigm:** Autonomous Multi-Agent Feedback Loop with 6 Hard Pass/Fail Rubric Checkpoints

This is a multi-agent AI educational content generator that runs completely locally. It uses LangGraph to orchestrate a pipeline where an AI generates a lesson, and a strict AI Evaluator grades it against a 6-point rubric. If it fails, the pipeline automatically forces the AI to revise and regenerate until it passes.

## 1. System Architecture & Design Choices

```text
                                  ┌───────────────────────────┐
                                  │   Target Persona Config   │
                                  │ (12th Grade Indian Student│
                                  │ Non-English Background)   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
┌──────────────────┐               ┌───────────────────────────┐
│   Input Topic    │ ────────────► │      Lesson Generator     │ ◄───────┐
│ (e.g., Intro RAG)│               │   (Few-shot + Memory)     │         │
└──────────────────┘               └─────────────┬─────────────┘         │
                                                │                        │
                                                ▼ (Draft Lesson)         │
                                   ┌───────────────────────────┐         │ Feedback &
                                   │  Deterministic & LLM Eval │         │ Rejection Log
                                   │  (6 Hard Pass/Fail Checks)│         │ (Max 2 retries)
                                   └─────────────┬─────────────┘         │
                                                │                        │
                                          [All Pass?]                    │
                                         /           \                   │
                                   (YES)/             \(NO)              │
                                       ▼               ▼                 │
                         ┌───────────────────┐   ┌───────────────────────┴───┐
                         │   Final Lesson    │   │ Has Retry Limit Reached?  │
                         │         +         │   │  - NO: Regenerate w/ Diff │
                         │ Full Audit Trail  │   │  - YES: Fail Gracefully   │
                         └───────────────────┘   └───────────────────────────┘
                                   ▲
                                   │ Update Failure Patterns
                         ┌─────────┴─────────┐
                         │ Persistent Memory │
                         │ (SQLite / JSON)   │
                         └───────────────────┘
 ``` 

## Key Design Choices & Trade-offs

1. Why LangGraph over Linear Chains / Vanilla Loops?

Explicit Cyclic State Machine: Evaluation and conditional regeneration require stateful branching. LangGraph guarantees deterministic edge routing.

State Transparency: Every iteration preserves previous revisions, diffs, and checkpoint scores in the state graph.

Guaranteed Termination: A strict retry_count <= MAX_RETRIES check prevents infinite loops.

2. Hard Pass/Fail Rubric (No Partial Credit):

Partial scores (e.g., 3.5/5) introduce ambiguity into automated shipping decisions.

We enforce 6 strict Boolean checkpoints: grounding_accuracy, persona_language, pedagogical_analogy, zero_unexplained_jargon, core_coverage, and pedagogical_flow.

3. Persistent Memory & Self-Evolution:

PersistentFailureMemory logs failure types across historical runs.

Common failure modes are dynamically pulled into the system prompt as negative constraints ("Anti-patterns learned from past runs"), making the system self-evolving. 

2. Project Directory Structure
rag_content_engine/
├── requirements.txt
├── schemas.py
├── memory.py
├── evaluator.py
├── generator.py
├── graph.py
├── model_factory.py
├── run_pipeline.py
└── README.md

3. Tech Stack
Framework: LangGraph & LangChain

LLM Provider: Local Execution via Ollama (or free cloud via Groq)

Model: Llama 3.1 (8B)

4. Setup Instructions
 
Download and install Ollama.

Pull the model in your terminal:ollama run llama3.1:8b

Install Python dependencies:pip install -r requirements.txt

Set your environment variables:

Windows (PowerShell):$env:LLM_PROVIDER="ollama"$env:OLLAMA_MODEL="llama3.1:8b"

Run the pipeline:python run_pipeline.py


5. Engineering Trade-offs & Production Considerations
Model Choice: Evaluator utilizes temperature = 0.0 with structured outputs for strict adherence, while the Generator utilizes temperature = 0.7 for creative pedagogical phrasing.

Cost vs. Quality: The pipeline uses two-stage evaluation (fast deterministic Python regex/Flesch-Kincaid check first, LLM judge second).

Infinite loop guardrail: A strict max_retries = 2 hard cap is enforced.
