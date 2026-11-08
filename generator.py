from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import LessonContent, RejectionRecord
from memory import PersistentFailureMemory
from model_factory import get_chat_model

class LessonGenerator:
    def __init__(self):
        # Uses temperature 0.7 for creative explanation
        self.llm = get_chat_model(temperature=0.7)
        self.memory = PersistentFailureMemory()

    def generate(self, topic: str, deliberate_error: Optional[str] = None) -> LessonContent:
        anti_patterns = self.memory.get_learned_anti_patterns()

        system_prompt = f"""You are a Master AI Educator teaching 12th-grade students in India who attended regional-language schools.
Learner: Age 17-18, cleared 12th standard. Direct, simple, visual English. Relatable Indian life analogies.

HARD CRITERIA:
1. What, Why, and How (all 3 must be crystal clear).
2. Analogy First: Use an open-book exam analogy (Closed-book student = normal AI; Open-book student = RAG AI).
3. Zero Unexplained Jargon: If you write 'LLM', immediately explain it ("Large Language Model - like a smart chatbot").
4. Clear 4-Step Mechanism:
   - Step 1: Storage / Indexing
   - Step 2: Library Search (Retrieve)
   - Step 3: Combining question + notes (Augment)
   - Step 4: Giving answer (Generate)
5. Interactive Check: 2 simple quiz questions at the end.

LEARNED SYSTEM ANTI-PATTERNS:
{anti_patterns}
"""
        user_prompt = f"Create a comprehensive beginner lesson on: {topic}"
        if deliberate_error == "unexplained_jargon":
            user_prompt += "\n[TEST MODE INSTRUCTION]: Intentionally use words like 'high-dimensional cosine vector embeddings' and 'k-NN dense retrieval' without explaining them at all, to test the evaluator."

        structured_llm = self.llm.with_structured_output(LessonContent)
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

    def regenerate(self, topic: str, current_lesson: LessonContent, rejection_records: List[RejectionRecord]) -> LessonContent:
        latest = rejection_records[-1]
        issues_breakdown = "\n".join([
            f"- FAILED: {cp}\n  Reason: {rs}\n  How to fix: {rem}"
            for cp, rs, rem in zip(latest.failed_checkpoints, latest.rejection_reasons, latest.remediation_prompts)
        ])

        system_prompt = """You are an expert AI curriculum refiner.
Your previous draft failed our strict quality audit. Surgically address every rejection reason.
Keep language simple, clear, and friendly for a 12th-grade student from India."""

        user_prompt = f"""TOPIC: {topic}
PREVIOUS DRAFT CONTENT:
{current_lesson.lesson_body_markdown}

AUDIT FAILURES THAT CAUSED REJECTION:
{issues_breakdown}

TASK: Rewrite the lesson completely so it clears 100% of these checkpoints."""

        structured_llm = self.llm.with_structured_output(LessonContent)
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
