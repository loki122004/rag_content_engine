"""Lesson Generator & Revision Agent tailored to the 12th-grade Indian student persona."""
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from schemas import LessonContent, RejectionRecord
from memory import PersistentFailureMemory


class LessonGenerator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.memory = PersistentFailureMemory()

    def generate(self, topic: str, deliberate_error: Optional[str] = None) -> LessonContent:
        anti_patterns = self.memory.get_learned_anti_patterns()

        system_prompt = f"""You are a Master AI Educator who excels at teaching complex computer science to 12th-grade students in India who went to regional-language medium schools.

YOUR TARGET LEARNER:
- Age 17-18, cleared 12th standard.
- Limited English vocabulary (prefers direct, simple, visual English; no flowery words).
- Aspires to build an AI career from scratch.
- Connects best with everyday Indian life analogies (school exams, cricket scoreboards, kirana store registers, library books).

HARD CRITERIA YOU MUST HIT:
1. What, Why, and How (all 3 must be crystal clear).
2. Analogy First: Use an open-book exam analogy (Closed-book student = normal AI; Open-book student with reference textbook = RAG AI).
3. Zero Unexplained Jargon: If you write 'LLM', immediately explain it ("Large Language Model - like a very smart chatbot trained on public text"). If you write 'Vector', explain it simply.
4. Clear Step-by-Step Mechanism:
   - Step 1: Library Search (Retrieve)
   - Step 2: Combining question + notes (Augment)
   - Step 3: Giving answer (Generate)
5. Interactive Check: Include 2 simple quiz questions at the end.

LEARNED SYSTEM ANTI-PATTERNS (Avoid these past mistakes):
{anti_patterns}
"""

        user_prompt = f"Create a comprehensive beginner lesson on: {topic}"

        if deliberate_error == "unexplained_jargon":
            user_prompt += "\n[TEST MODE INSTRUCTION]: Intentionally use words like 'high-dimensional cosine vector embeddings' and 'k-NN dense retrieval' without explaining them at all, to test the evaluator."
        elif deliberate_error == "missing_analogy":
            user_prompt += "\n[TEST MODE INSTRUCTION]: Do not use any analogies or real-world examples. Just define things mathematically."

        structured_llm = self.llm.with_structured_output(LessonContent)
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

    def regenerate(
        self,
        topic: str,
        current_lesson: LessonContent,
        rejection_records: List[RejectionRecord]
    ) -> LessonContent:
        latest = rejection_records[-1]

        issues_breakdown = "\n".join([
            f"- FAILED: {cp}\n  Reason: {rs}\n  How to fix: {rem}"
            for cp, rs, rem in zip(
                latest.failed_checkpoints,
                latest.rejection_reasons,
                latest.remediation_prompts
            )
        ])

        system_prompt = """You are an expert AI curriculum refiner.
Your previous draft failed our strict quality audit. You must revise the lesson to fix every single failed checkpoint.
Keep what worked, but surgically address every rejection reason.
Do not repeat past mistakes. Keep language simple, clear, and friendly for a 12th-grade student from India."""

        user_prompt = f"""TOPIC: {topic}

PREVIOUS DRAFT CONTENT:
{current_lesson.lesson_body_markdown}

AUDIT FAILURES THAT CAUSED REJECTION:
{issues_breakdown}

TASK:
Rewrite the lesson completely so it clears 100% of these checkpoints. Make sure all explanations are simple, analogies are crystal clear, and no jargon is left unexplained."""

        structured_llm = self.llm.with_structured_output(LessonContent)
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])