"""Rubric Evaluator enforcing 6 Hard Pass/Fail Checkpoints (Zero partial credit)."""
import re
import textstat
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from schemas import LessonContent, RubricEvaluation, CheckpointResult


class RubricEvaluator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)

    def evaluate(self, lesson: LessonContent) -> RubricEvaluation:
        # 1. Deterministic readability check (Flesch Kincaid Grade Level)
        fk_grade = textstat.flesch_kincaid_grade(lesson.lesson_body_markdown)
        readability_passed = fk_grade <= 8.5
        readability_reason = (
            f"Flesch-Kincaid Grade Level is {fk_grade:.1f} (Target <= 8.5 for 12th-grade Indian student)."
            if readability_passed
            else f"Language complexity too high: Flesch-Kincaid Grade is {fk_grade:.1f}. Target is <= 8.5."
        )

        # 2. LLM-based Hard Rubric Judge
        eval_prompt = f"""You are a strict QA auditor for beginner-level AI learning content.
Target Audience: A 12th-grade pass student from India from a non-English-medium school.
They know basic smartphone usage but NO programming, NO data science, and NO advanced math.

Evaluate the lesson below against these 6 HARD PASS/FAIL checkpoints.
There is NO PARTIAL CREDIT. If a single requirement of a checkpoint is missed, it FAILS (passed=False).

CHECKPOINTS:
1. 'grounding_accuracy':
   - Explains RAG accurately without false technical claims or hallucinations.
   - Clearly explains that RAG retrieves factual context from an external database and feeds it to an LLM.

2. 'beginner_friendly_language':
   - Grade level matches target student.
   - Sentences are short and active.
   - Tone is encouraging and relatable.
   - Automated Readability Check: {readability_reason} (Take this into strong consideration).

3. 'teaches_by_example':
   - MUST contain an intuitive, everyday real-life analogy (e.g., student taking an open-book exam vs closed-book).
   - The analogy must be explicitly mapped to: External Documents, Retriever, and Generator.

4. 'zero_unexplained_jargon':
   - Every technical buzzword (LLM, Embeddings, Vector Database, Prompt, Hallucination, Retrieval, Context Window) MUST be immediately defined in simple words upon first appearance.
   - If even ONE jargon word is used without a beginner explanation, this checkpoint FAILS.

5. 'core_coverage':
   - Must cover ALL three core questions:
     a) What is RAG?
     b) Why does it matter? (solves outdated training data & hallucinations)
     c) How does it work? (Step 1: Store/Index, Step 2: Search/Retrieve, Step 3: Augment/Prompt, Step 4: Generate).

6. 'coherent_teaching_flow':
   - Logical progression: Hook/Problem -> Relatable Analogy -> Architecture/Steps -> Practical Example -> Quiz.
   - No jarring leaps in logic.

LESSON TO EVALUATE:
---
TITLE: {lesson.title}
CONTENT:
{lesson.lesson_body_markdown}
---
"""
        structured_eval_llm = self.llm.with_structured_output(RubricEvaluation)
        eval_result: RubricEvaluation = structured_eval_llm.invoke([
            SystemMessage(content="You are an unyielding, rigorous educational quality evaluator."),
            HumanMessage(content=eval_prompt)
        ])

        # Enforce deterministic override if readability failed outright
        for cp in eval_result.checkpoints:
            if cp.name == "beginner_friendly_language" and not readability_passed:
                cp.passed = False
                cp.reason += f" [Automated Metric Alert: {readability_reason}]"
                cp.remediation_guidance = "Simplify vocabulary and split long compound sentences."

        # Compute overall pass: strictly all True
        eval_result.overall_passed = all(cp.passed for cp in eval_result.checkpoints)
        return eval_result