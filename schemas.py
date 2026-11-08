from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CheckpointResult(BaseModel):
    name: str = Field(description="Name of the rubric checkpoint")
    passed: bool = Field(description="Strict boolean pass/fail status")
    reason: str = Field(description="Concrete reason why it passed or failed")
    remediation_guidance: Optional[str] = Field(
        default=None, description="Actionable instruction on how to fix this on retry"
    )

class RubricEvaluation(BaseModel):
    checkpoints: List[CheckpointResult] = Field(description="Results for all 6 hard pass/fail checkpoints")
    overall_passed: bool = Field(description="True ONLY if 100% of checkpoints pass")
    summary: str = Field(description="High-level feedback summary")

class LessonContent(BaseModel):
    topic: str
    title: str
    target_audience: str
    lesson_body_markdown: str = Field(description="The full lesson content formatted in Markdown")
    key_takeaways: List[str]
    quick_quiz: List[Dict[str, Any]] = Field(description="2-3 simple conceptual check questions")

class RejectionRecord(BaseModel):
    attempt_number: int
    failed_checkpoints: List[str]
    rejection_reasons: List[str]
    remediation_prompts: List[str]

class PipelineState(BaseModel):
    topic: str
    current_attempt: int = 1
    max_retries: int = 2
    current_lesson: Optional[LessonContent] = None
    latest_evaluation: Optional[RubricEvaluation] = None
    rejection_history: List[RejectionRecord] = Field(default_factory=list)
    status: str = "INITIAL"
    injected_error: Optional[str] = None
