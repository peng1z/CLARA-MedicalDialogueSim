"""Performance analytics, scoring, and expert benchmarking.

Includes:
- compute_score: original weighted score
- compute_reasoning_quality_score: VietMed-style 0-100 scoring with A-F grades
- compare_to_expert_baseline: compare trainee metrics against MIMIC expert baselines
- save_profile_update: persist student progress
"""
from typing import Dict, List
import json
from pathlib import Path

_PROFILE_PATH = Path(__file__).resolve().parents[1] / "data" / "student_profiles.json"

# --- Expert baselines from VietMed.ipynb MIMIC-IV analysis ---
EXPERT_BASELINES = {
    "total_documents": 500,
    "total_entities": 4447,
    "avg_entities_per_doc": 8.89,
    "questions": 33,
    "hypotheses": 85,
    "confidence_mean": 0.9151,
    "high_confidence_pct": 73.6,
    "entity_types": {"problem": 2812, "treatment": 1401, "test": 234},
}


def compute_score(diagnostic_feedback: Dict, communication_feedback: Dict) -> Dict:
    """Original weighted score: 70% diagnostic coverage, 30% open question ratio."""
    diag_score = diagnostic_feedback.get("coverage", 0.0)
    comm_score = communication_feedback.get("open_ratio", 0.0)
    total = 0.7 * diag_score + 0.3 * comm_score

    peer_avg = 0.72
    peer_sd = 0.08

    return {"score": round(total, 3), "peer_avg": peer_avg, "peer_sd": peer_sd}


def compute_reasoning_quality_score(
    questions: List[str],
    hypotheses: List[str],
    total_entities: int,
    confidence_scores: List[float],
) -> Dict:
    """
    Comprehensive reasoning quality score (0-100) from VietMed.ipynb integration.

    Components:
    - Volume (0-30): number of reasoning elements (questions + hypotheses)
    - Diversity (0-25): variety in reasoning patterns
    - Confidence (0-25): average NER confidence
    - Density (0-20): reasoning elements per entity

    Returns dict with component scores, overall score, and letter grade.
    """
    # Volume score (0-30 points)
    total_reasoning = len(questions) + len(hypotheses)
    volume_score = min(30, total_reasoning * 0.5)

    # Diversity score (0-25 points)
    unique_q = len(set(q[:20] for q in questions)) if questions else 0
    unique_h = len(set(h[:20] for h in hypotheses)) if hypotheses else 0
    diversity_score = min(25, (unique_q + unique_h) * 1.5)

    # Confidence score (0-25 points)
    if confidence_scores:
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        confidence_score = avg_conf * 25
    else:
        confidence_score = 0.0

    # Density score (0-20 points)
    if total_entities > 0:
        reasoning_density = total_reasoning / total_entities
        density_score = min(20, reasoning_density * 10)
    else:
        density_score = 0.0

    overall = volume_score + diversity_score + confidence_score + density_score

    if overall >= 80:
        grade = "A"
    elif overall >= 65:
        grade = "B"
    elif overall >= 50:
        grade = "C"
    elif overall >= 35:
        grade = "D"
    else:
        grade = "F"

    return {
        "volume_score": round(volume_score, 1),
        "diversity_score": round(diversity_score, 1),
        "confidence_score": round(confidence_score, 1),
        "density_score": round(density_score, 1),
        "overall_score": round(overall, 1),
        "grade": grade,
    }


def compare_to_expert_baseline(
    trainee_questions: int,
    trainee_hypotheses: int,
    trainee_entities: int,
    trainee_confidence: float,
) -> Dict:
    """
    Compare trainee metrics against MIMIC-IV expert baselines (from VietMed.ipynb).

    Returns dict with expert metrics, trainee metrics, and advantage ratios.
    """
    exp = EXPERT_BASELINES
    q_ratio = exp["questions"] / max(trainee_questions, 1)
    h_ratio = exp["hypotheses"] / max(trainee_hypotheses, 1)
    avg_advantage = (q_ratio + h_ratio) / 2

    return {
        "expert": {
            "questions": exp["questions"],
            "hypotheses": exp["hypotheses"],
            "entities": exp["total_entities"],
            "confidence": exp["confidence_mean"],
        },
        "trainee": {
            "questions": trainee_questions,
            "hypotheses": trainee_hypotheses,
            "entities": trainee_entities,
            "confidence": trainee_confidence,
        },
        "expert_advantage": {
            "questions_ratio": round(q_ratio, 1),
            "hypotheses_ratio": round(h_ratio, 1),
            "overall": round(avg_advantage, 1),
        },
    }


def save_profile_update(student_id: str, updates: Dict) -> None:
    """Append updates to profile store (simple JSON list). Create file if missing."""
    try:
        data = {"profiles": {}}
        if _PROFILE_PATH.exists():
            with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        profiles = data.get("profiles", {})
        profile = profiles.get(student_id, {"id": student_id, "history": []})
        profile["history"].append(updates)
        profiles[student_id] = profile
        with open(_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump({"profiles": profiles}, f, indent=2)
    except Exception:
        pass
