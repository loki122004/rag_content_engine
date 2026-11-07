"""LangGraph orchestrator tying Generator -> Evaluator -> Conditional Router -> Memory."""
from langgraph.graph import StateGraph, END
from schemas import PipelineState, RejectionRecord
from generator import LessonGenerator
from evaluator import RubricEvaluator
from memory import PersistentFailureMemory


class ContentPipeline:
    def __init__(self, model_name: str = "gpt-4o"):
        self.generator = LessonGenerator(model_name=model_name)
        self.evaluator = RubricEvaluator(model_name=model_name)
        self.memory = PersistentFailureMemory()
        self.workflow = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PipelineState)

        graph.add_node("generate_draft", self._node_generate)
        graph.add_node("evaluate_draft", self._node_evaluate)
        graph.add_node("revise_draft", self._node_revise)

        graph.set_entry_point("generate_draft")
        graph.add_edge("generate_draft", "evaluate_draft")

        graph.add_conditional_edges(
            "evaluate_draft",
            self._route_decision,
            {
                "pass": END,
                "revise": "revise_draft",
                "max_retries_exceeded": END
            }
        )
        graph.add_edge("revise_draft", "evaluate_draft")

        return graph.compile()

    def _node_generate(self, state: PipelineState) -> dict:
        lesson = self.generator.generate(
            topic=state.topic,
            deliberate_error=state.injected_error
        )
        return {"current_lesson": lesson, "status": "EVALUATING"}

    def _node_evaluate(self, state: PipelineState) -> dict:
        evaluation = self.evaluator.evaluate(state.current_lesson)

        if evaluation.overall_passed:
            self.memory.record_success()
            return {"latest_evaluation": evaluation, "status": "PASSED"}
        else:
            failed_cps = [cp.name for cp in evaluation.checkpoints if not cp.passed]
            reasons = [cp.reason for cp in evaluation.checkpoints if not cp.passed]
            remediations = [cp.remediation_guidance or "Fix issue." for cp in evaluation.checkpoints if not cp.passed]

            record = RejectionRecord(
                attempt_number=state.current_attempt,
                failed_checkpoints=failed_cps,
                rejection_reasons=reasons,
                remediation_prompts=remediations
            )

            self.memory.record_failures(failed_cps, reasons)

            new_history = list(state.rejection_history)
            new_history.append(record)

            return {
                "latest_evaluation": evaluation,
                "rejection_history": new_history,
                "status": "REJECTED"
            }

    def _node_revise(self, state: PipelineState) -> dict:
        revised_lesson = self.generator.regenerate(
            topic=state.topic,
            current_lesson=state.current_lesson,
            rejection_records=state.rejection_history
        )
        return {
            "current_lesson": revised_lesson,
            "current_attempt": state.current_attempt + 1,
            "injected_error": None,
            "status": "EVALUATING"
        }

    def _route_decision(self, state: PipelineState) -> str:
        if state.latest_evaluation and state.latest_evaluation.overall_passed:
            return "pass"
        if state.current_attempt >= state.max_retries:
            return "max_retries_exceeded"
        return "revise"

    def run(self, topic: str, deliberate_error: str = None) -> PipelineState:
        initial_state = PipelineState(
            topic=topic,
            injected_error=deliberate_error
        )
        return self.workflow.invoke(initial_state)