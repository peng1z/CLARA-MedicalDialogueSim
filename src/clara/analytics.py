"""Performance analytics, scoring, and expert benchmarking.

Includes:
- compute_score: original weighted score
- compute_reasoning_quality_score: VietMed-style 0-100 scoring with A-F grades
- compare_to_expert_baseline: compare trainee metrics against expert baselines
- save_profile_update: persist student progress

Two separate baseline sets:
  MIMIC_BASELINES  — derived from 999 MIMIC-IV discharge notes via
                     scripts/compute_mimic_baselines.py. Used for
                     concept/entity distribution and peer scoring.
  VIETMED_BASELINES — conversational question/hypothesis counts from
                      VietMed.ipynb analysis of clinical dialogues.
                      Used when comparing trainee session transcripts.
"""
from typing import Dict, List
import json
from pathlib import Path

_PROFILE_PATH   = Path(__file__).resolve().parents[1] / "data" / "student_profiles.json"
_BASELINES_PATH = Path(__file__).resolve().parents[1] / "data" / "mimic_baselines.json"

# ---------------------------------------------------------------------------
# MIMIC-IV baselines — computed from 999 discharge notes
# (scripts/compute_mimic_baselines.py, prose pattern set)
# ---------------------------------------------------------------------------
MIMIC_BASELINES = {
    "total_documents":    999,
    "total_entities":     4395,
    "avg_entities_per_doc": 4.40,
    "questions":          3.45,   # avg questions per note (prose patterns)
    "hypotheses":         1.04,   # avg hypotheses per note (prose patterns)
    "confidence_mean":    0.9159, # validated with samrawal/bert-base-uncased_clinical-ner
    "high_confidence_pct": 73.9,
    "entity_types":       {"problem": 5595, "treatment": 2978, "test": 478},
    "score_distribution": {
        "mean":  42.79,
        "stdev":  9.15,
        "min":   21.40,
        "p25":   35.70,
        "p50":   41.90,
        "p75":   49.50,
        "p90":   56.00,
        "max":   70.20,
    },
}

# ---------------------------------------------------------------------------
# VietMed conversational baselines — from VietMed.ipynb analysis of clinical
# dialogues (expert clinicians during patient interviews)
# ---------------------------------------------------------------------------
VIETMED_BASELINES = {
    "questions":  33,     # expert clinician asks ~33 questions per session
    "hypotheses": 85,     # expert clinician generates ~85 hypotheses per session
    "confidence_mean": 0.9151,
    "high_confidence_pct": 73.6,
    "entity_types": {"problem": 2812, "treatment": 1401, "test": 234},
}

# Default alias — conversational sessions use VietMed, prose notes use MIMIC
EXPERT_BASELINES = VIETMED_BASELINES


def _load_mimic_baselines() -> Dict:
    """Load mimic_baselines.json if available, fall back to hardcoded values."""
    if _BASELINES_PATH.exists():
        try:
            with open(_BASELINES_PATH) as f:
                data = json.load(f)
            # merge into MIMIC_BASELINES (file is authoritative)
            return {**MIMIC_BASELINES, **{k: v for k, v in data.items() if not k.startswith("_")}}
        except Exception:
            pass
    return MIMIC_BASELINES


def compute_score(diagnostic_feedback: Dict, communication_feedback: Dict) -> Dict:
    """Original weighted score: 70% diagnostic coverage, 30% open question ratio.

    peer_avg and peer_sd are derived from MIMIC-IV score distribution
    (converted from 0-100 scale to 0-1 to match coverage/ratio scale).
    """
    diag_score = diagnostic_feedback.get("coverage", 0.0)
    comm_score = communication_feedback.get("open_ratio", 0.0)
    total = 0.7 * diag_score + 0.3 * comm_score

    # Derived from MIMIC_BASELINES score_distribution (mean=42.79, stdev=9.15)
    # computed from 999 MIMIC-IV notes with transformer NER
    peer_avg = 0.4279
    peer_sd  = 0.0915

    return {"score": round(total, 3), "peer_avg": peer_avg, "peer_sd": peer_sd}


def compute_reasoning_quality_score(
    questions: List[str],
    hypotheses: List[str],
    total_entities: int,
    confidence_scores: List[float],
) -> Dict:
    """Comprehensive reasoning quality score (0-100) from VietMed.ipynb integration.

    Components:
    - Volume    (0-30): number of reasoning elements (questions + hypotheses)
    - Diversity (0-25): variety in reasoning patterns
    - Confidence(0-25): average NER confidence
    - Density   (0-20): reasoning elements per entity

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
        "volume_score":     round(volume_score, 1),
        "diversity_score":  round(diversity_score, 1),
        "confidence_score": round(confidence_score, 1),
        "density_score":    round(density_score, 1),
        "overall_score":    round(overall, 1),
        "grade":            grade,
    }


def compare_to_expert_baseline(
    trainee_questions: int,
    trainee_hypotheses: int,
    trainee_entities: int,
    trainee_confidence: float,
    mode: str = "dialogue",
) -> Dict:
    """Compare trainee metrics against expert baselines.

    Args:
        mode: "dialogue" uses VietMed conversational baselines (default,
              for trainee session transcripts); "prose" uses MIMIC-IV
              discharge note baselines (for clinical text analysis).

    Returns dict with expert metrics, trainee metrics, and advantage ratios.
    """
    if mode == "prose":
        exp = _load_mimic_baselines()
    else:
        exp = VIETMED_BASELINES

    q_ratio  = exp["questions"]  / max(trainee_questions,  1)
    h_ratio  = exp["hypotheses"] / max(trainee_hypotheses, 1)
    avg_advantage = (q_ratio + h_ratio) / 2

    return {
        "mode": mode,
        "expert": {
            "total_questions":  exp["questions"],
            "total_hypotheses": exp["hypotheses"],
            "total_entities":   exp.get("total_entities", 0),
            "confidence":       exp.get("confidence_mean", 0.0),
        },
        "trainee": {
            "total_questions":  trainee_questions,
            "total_hypotheses": trainee_hypotheses,
            "total_entities":   trainee_entities,
            "confidence":       trainee_confidence,
        },
        "expert_advantage": {
            "questions_ratio":  round(q_ratio, 2),
            "hypotheses_ratio": round(h_ratio, 2),
            "overall":          round(avg_advantage, 2),
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
