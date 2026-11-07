"""Driver script demonstrating end-to-end execution, logging, and deliberate error catching."""
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from graph import ContentPipeline

console = Console()

def run_test(deliberate_error=None):
    console.print(Panel(
        f"[bold cyan]Running Self-Evaluating Lesson Generator[/bold cyan]\n"
        f"Topic: [yellow]Introduction to RAG[/yellow]\n"
        f"Deliberate Error Injection: [magenta]{deliberate_error or 'None (Standard Mode)'}[/magenta]",
        expand=False
    ))

    pipeline = ContentPipeline()
    result = pipeline.run(
        topic="Introduction to RAG (Retrieval-Augmented Generation)",
        deliberate_error=deliberate_error
    )

    if result.rejection_history:
        console.print("\n[bold red]─── REJECTION AUDIT LOG ───[/bold red]")
        for record in result.rejection_history:
            table = Table(title=f"Attempt #{record.attempt_number} Failed Checks")
            table.add_column("Checkpoint", style="red bold")
            table.add_column("Failure Reason", style="white")
            table.add_column("Remediation", style="green")

            for cp, rs, rem in zip(record.failed_checkpoints, record.rejection_reasons, record.remediation_prompts):
                table.add_row(cp, rs, rem)
            console.print(table)
    else:
        console.print("[green]✓ First attempt passed all rubric checkpoints with zero rejections![/green]")

    console.print(f"\n[bold]Final Status:[/bold] [green]{result.status}[/green] (Attempt #{result.current_attempt})")

    if result.latest_evaluation:
        eval_table = Table(title="Final Rubric Evaluation")
        eval_table.add_column("Checkpoint", style="cyan")
        eval_table.add_column("Status", justify="center")
        eval_table.add_column("Reason", style="dim")

        for cp in result.latest_evaluation.checkpoints:
            status_str = "[green]PASS[/green]" if cp.passed else "[red]FAIL[/red]"
            eval_table.add_row(cp.name, status_str, cp.reason)
        console.print(eval_table)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "standard"
    if mode == "test_error":
        run_test(deliberate_error="unexplained_jargon")
    else:
        run_test()