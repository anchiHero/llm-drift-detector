# LLM Drift Detector

Detects semantic drift in LLM outputs by running a fixed set of "canary" prompts
on a schedule, embedding the responses, and comparing similarity to a baseline.

## Why this exists

LLM providers update their models silently — the same API endpoint today
may behave differently than it did last week. This project continuously
monitors those changes so you catch regressions before customers do.

## How it works

1. **Baseline** — send each canary prompt N times, average the embeddings
2. **Daily check** — send each prompt once, embed the response
3. **Compare** — cosine similarity between today's embedding and the baseline
4. **Alert** — if similarity drops below threshold, open a GitHub issue

## Drift severity thresholds

| Similarity   | Severity   | Action                      |
|--------------|------------|-----------------------------|
| ≥ 0.90       | none       | All good                    |
| 0.75 – 0.90  | mild       | Monitor                     |
| 0.55 – 0.75  | moderate   | Investigate                 |
| < 0.55       | severe     | Alert + review baseline     |

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/llm-drift-detector
cd llm-drift-detector
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY