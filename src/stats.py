"""Drift statistics: compute summaries and detect anomalies."""
import numpy as np

def drift_severity(similarity: float) -> str:
    """Categorize drift based on cosine similarity to baseline."""
    if similarity >= 0.90:
        return "none"
    elif similarity >= 0.75:
        return "mild"
    elif similarity >= 0.55:
        return "moderate"
    else:
        return "severe"


def rolling_drift_score(similarities: list[float], window: int = 5) -> float:
    """Mean of the last N similarity scores — smooths noise."""
    recent = similarities[-window:] if len(similarities) >= window else similarities
    return float(np.mean(recent)) if recent else 1.0


def is_anomaly(current: float, history: list[float], z_threshold: float = 2.0) -> bool:
    """Flag if current similarity is >z_threshold std devs below the mean."""
    if len(history) < 5:
        return False
    mean, std = np.mean(history), np.std(history)
    if std == 0:
        return False
    z = (current - mean) / std
    return z < -z_threshold