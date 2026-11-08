import textstat
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import LessonContent, RubricEvaluation
from model_factory import get_chat_model

class RubricEvaluator:
    def __init__(self):
        # Uses temperature 0.0 for deterministic evaluation
        self.llm = get_chat_model(temperature=0.0)

    def evaluate(self, lesson: LessonContent) -> RubricEvaluation:
        # Deterministic readability check
        fk_grade = textstat.flesch_kincaid_grade(lesson.lesson_body_markdown)
        readability_passed = fk_grade <= 8.5
        readability_reason = (
            f"Flesch-Kincaid Grade Level is {fk_grade:.1f} (Target <= 8.5 for 12th-grade Indian student)."
            if readability_passed
            else f"Language complexity too high: Flesch-Kincaid Grade is {fk_grade:.1f}. Target is <= 8.5."
        )

        eval_prompt = f"""You are a strict QA auditor for beginner-level AI learning content.
Target Audience: A 12th-grade pass student from India from a non-English-medium school.
They know basic smartphone usage but NO programming, NO data science, and NO advanced math.

Evaluate the lesson below against these 6 HARD PASS/FAIL checkpoints.
There is NO PARTIAL CREDIT. If a single requirement of a checkpoint is missed, it FAILS (passed=False).

CHECKPOINTS:
1. 'grounding_accuracy': Explains RAG accurately without false technical claims or hallucinations.
2. 'beginner_friendly_language': Grade level matches target student. Readability metric: {readability_reason}.
3. 'teaches_by_example': MUST contain an intuitive everyday analogy (e.g. open-book exam vs closed-book).
4. 'zero_unexplained_jargon': Every technical buzzword (LLM, Embeddings, Vectors, Hallucination) MUST be defined immediately upon first appearance.
5. 'core_coverage': Must cover What is RAG, Why it matters (outdated data + hallucinations), and How it works (4 steps).
6. 'coherent_teaching_flow': Smooth progression: Hook -> Analogy -> 4 Steps -> Practical Example -> Quiz.

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

        eval_result.overall_passed = all(cp.passed for cp in eval_result.checkpoints)
        return eval_result
