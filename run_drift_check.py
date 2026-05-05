"""Entry point: initialize baseline or run a drift check."""
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from src.detector import create_baseline, check_drift, HISTORY_FILE
from src.stats import is_anomaly
import pandas as pd

load_dotenv()
console = Console()

PROMPTS = "prompts/canary_prompts.json"
MODEL = "claude-sonnet-4-5"
#MODEL = "claude-haiku-4-5"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Create baseline")
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt for baseline")
    args = parser.parse_args()

    if args.init:
        create_baseline(PROMPTS, MODEL, runs_per_prompt=args.runs)
        return

    console.print(f"[bold cyan]Running drift check against:[/] {MODEL}\n")
    df = check_drift(PROMPTS, MODEL)

    # Load full history for anomaly detection
    history = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()

    table = Table(title="Drift Check Results")
    for col in ["prompt_id", "category", "similarity", "severity", "anomaly?"]:
        table.add_column(col)

    severity_colors = {"none": "green", "mild": "yellow", "moderate": "orange1", "severe": "red"}
    for _, row in df.iterrows():
        prior = history[history["prompt_id"] == row["prompt_id"]]["similarity_to_baseline"].tolist()[:-1]
        anomaly = is_anomaly(row["similarity_to_baseline"], prior)
        color = severity_colors.get(row["severity"], "white")
        table.add_row(
            row["prompt_id"], row["category"],
            f"{row['similarity_to_baseline']:.3f}",
            f"[{color}]{row['severity']}[/{color}]",
            "⚠️ YES" if anomaly else "—"
        )
    console.print(table)

    avg_sim = df["similarity_to_baseline"].mean()
    severe_count = (df["severity"] == "severe").sum()
    console.print(f"\n[bold]Mean similarity:[/] {avg_sim:.3f}")
    console.print(f"[bold]Severe drift cases:[/] {severe_count}/{len(df)}")

    if severe_count > 0 or avg_sim < 0.80:
        console.print("\n[bold red]⚠️ Drift alert: review responses above[/]")
        exit(1)  # non-zero exit for CI/CD alerting

if __name__ == "__main__":
    main()