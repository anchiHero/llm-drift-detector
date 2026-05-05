"""Core drift detection: run canary prompts and compare against baseline."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from anthropic import Anthropic
import pandas as pd
import numpy as np
from src.embeddings import embed, cosine_similarity
from src.stats import drift_severity, is_anomaly

DATA_DIR = Path("data")
BASELINE_FILE = DATA_DIR / "baseline.json"
HISTORY_FILE = DATA_DIR / "history.csv"


def load_prompts(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["prompts"]


def call_model(client: Anthropic, model: str, prompt: str) -> str:
    msg = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def create_baseline(prompts_path: str, model: str, runs_per_prompt: int = 3):
    """Run each prompt N times and store the average embedding as the baseline."""
    client = Anthropic()
    prompts = load_prompts(prompts_path)
    baseline = {"model": model, "created_at": datetime.now(timezone.utc).isoformat(), "prompts": {}}

    print(f"Building baseline ({runs_per_prompt} runs per prompt)...")
    for p in prompts:
        print(f"  {p['id']}...", end=" ", flush=True)
        responses, embeddings = [], []
        for _ in range(runs_per_prompt):
            r = call_model(client, model, p["prompt"])
            responses.append(r)
            embeddings.append(embed(r))
        avg_emb = np.mean(embeddings, axis=0)
        baseline["prompts"][p["id"]] = {
            "prompt": p["prompt"],
            "category": p["category"],
            "baseline_embedding": avg_emb.tolist(),
            "sample_responses": responses
        }
        print("✓")

    DATA_DIR.mkdir(exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nBaseline saved: {BASELINE_FILE}")


def check_drift(prompts_path: str, model: str) -> pd.DataFrame:
    """Run canary prompts today and compare to baseline."""
    if not BASELINE_FILE.exists():
        raise FileNotFoundError("No baseline found. Run with --init first.")

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    client = Anthropic()
    prompts = load_prompts(prompts_path)
    run_time = datetime.now(timezone.utc).isoformat()
    results = []

    for p in prompts:
        if p["id"] not in baseline["prompts"]:
            continue
        start = time.time()
        try:
            response = call_model(client, model, p["prompt"])
            error = None
        except Exception as e:
            response, error = "", str(e)

        latency = time.time() - start
        similarity = 0.0
        if response:
            current_emb = embed(response)
            baseline_emb = np.array(baseline["prompts"][p["id"]]["baseline_embedding"])
            similarity = cosine_similarity(current_emb, baseline_emb)

        results.append({
            "timestamp": run_time,
            "prompt_id": p["id"],
            "category": p["category"],
            "model": model,
            "response": response,
            "error": error,
            "similarity_to_baseline": round(similarity, 4),
            "severity": drift_severity(similarity),
            "latency_sec": round(latency, 2),
        })

    df = pd.DataFrame(results)
    # Append to history
    DATA_DIR.mkdir(exist_ok=True)
    if HISTORY_FILE.exists():
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    return df