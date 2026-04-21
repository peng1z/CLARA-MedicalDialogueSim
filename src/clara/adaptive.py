"""Adaptive learning insights: update student profile and recommend next cases.

Includes:
- generate_adaptive_insights: basic gap detection + case recommendation
- generate_personalized_feedback: VietMed-style comprehensive feedback with
  strengths, improvement areas, expert examples, and learning pathways
- Optional LLM integration for intelligent recommendations.
"""
from typing import Dict, List, Optional, Any
from .analytics import save_profile_update, EXPERT_BASELINES

try:
    from .llm import LLMClient
    _LLM_AVAILABLE = True
except (ImportError, Exception):
    _LLM_AVAILABLE = False
    LLMClient = Any  # type: ignore


def generate_adaptive_insights(student_id: str, diagnostic_feedback: Dict, communication_feedback: Dict) -> Dict:
    gaps = diagnostic_feedback.get("missing", [])
    insights = {"gaps": gaps, "comm_open_ratio": communication_feedback.get("open_ratio", 0.0)}

    # Save to profile
    save_profile_update(student_id, insights)

    # Recommend next case: if cardiac gaps, recommend a cardiac-focused case
    next_case = "general_clinical_case"
    cardiac_markers = set(["chest_pain", "radiation", "shortness_of_breath", "diaphoresis"])
    if cardiac_markers.intersection(set(gaps)):
        next_case = "cardiac_chest_pain_case"

    return {"insights": insights, "recommended_next_case": next_case}


def generate_personalized_feedback(
    trainee_score: Dict,
    expert_score: Dict,
    trainee_questions: List[str],
    trainee_hypotheses: List[str],
    expert_questions: Optional[List[str]] = None,
    expert_hypotheses: Optional[List[str]] = None,
) -> Dict:
    """
    Generate personalized feedback comparing trainee to expert baselines.
    Ported from VietMed.ipynb's generate_personalized_feedback().

    Args:
        trainee_score: dict from compute_reasoning_quality_score() for trainee
        expert_score: dict from compute_reasoning_quality_score() for expert
        trainee_questions: list of questions the trainee asked
        trainee_hypotheses: list of hypotheses the trainee formed
        expert_questions: example expert questions (loaded from KB if None)
        expert_hypotheses: example expert hypotheses (loaded from KB if None)

    Returns:
        Dict with overall_assessment, strengths, improvement_areas,
        specific_recommendations, expert_examples, and learning_pathway.
    """
    # Load expert examples from KB if not provided
    if expert_questions is None or expert_hypotheses is None:
        try:
            from .semantic import load_kb
            kb = load_kb()
            sc = kb.get("scenario_chest_pain", {})
            patterns = sc.get("expert_reasoning_patterns", {})
            expert_questions = expert_questions or patterns.get("expert_questions", [])
            expert_hypotheses = expert_hypotheses or patterns.get("expert_hypotheses", [])
        except Exception:
            expert_questions = expert_questions or []
            expert_hypotheses = expert_hypotheses or []

    feedback = {
        "overall_assessment": "",
        "strengths": [],
        "improvement_areas": [],
        "specific_recommendations": [],
        "expert_examples": [],
        "learning_pathway": {},
    }

    # Overall assessment based on score ratio
    t_score = trainee_score.get("overall_score", 0)
    e_score = expert_score.get("overall_score", 1)
    score_ratio = t_score / max(e_score, 1)

    if score_ratio >= 0.8:
        feedback["overall_assessment"] = (
            "Excellent clinical reasoning skills approaching expert level. "
            f"Current performance: {t_score}/100 ({score_ratio:.0%} of expert level)."
        )
    elif score_ratio >= 0.6:
        feedback["overall_assessment"] = (
            "Good clinical reasoning with room for systematic improvement. "
            f"Current performance: {t_score}/100 ({score_ratio:.0%} of expert level)."
        )
    elif score_ratio >= 0.4:
        feedback["overall_assessment"] = (
            "Developing clinical reasoning skills requiring focused practice. "
            f"Current performance: {t_score}/100 ({score_ratio:.0%} of expert level)."
        )
    else:
        feedback["overall_assessment"] = (
            "Early-stage clinical reasoning needing comprehensive development. "
            f"Current performance: {t_score}/100 ({score_ratio:.0%} of expert level)."
        )

    # Identify strengths
    if trainee_score.get("volume_score", 0) / max(expert_score.get("volume_score", 1), 1) >= 0.7:
        feedback["strengths"].append("Strong questioning and hypothesis generation volume")
    if trainee_score.get("diversity_score", 0) / max(expert_score.get("diversity_score", 1), 1) >= 0.7:
        feedback["strengths"].append("Good variety in reasoning approaches")
    if trainee_score.get("confidence_score", 0) / max(expert_score.get("confidence_score", 1), 1) >= 0.8:
        feedback["strengths"].append("Accurate medical terminology usage")
    if trainee_score.get("density_score", 0) / max(expert_score.get("density_score", 1), 1) >= 0.7:
        feedback["strengths"].append("Appropriate reasoning depth per medical concept")

    if not feedback["strengths"]:
        feedback["strengths"].append("Willingness to engage with clinical scenarios")

    # Identify improvement areas + specific recommendations
    vol_gap = expert_score.get("volume_score", 0) - trainee_score.get("volume_score", 0)
    div_gap = expert_score.get("diversity_score", 0) - trainee_score.get("diversity_score", 0)
    conf_gap = expert_score.get("confidence_score", 0) - trainee_score.get("confidence_score", 0)
    dens_gap = expert_score.get("density_score", 0) - trainee_score.get("density_score", 0)

    if vol_gap > 5:
        feedback["improvement_areas"].append(
            f"Volume: Generate more questions and hypotheses (+{vol_gap:.1f} points needed)"
        )
        feedback["specific_recommendations"].append(
            "Practice generating 3-5 differential diagnoses for each case"
        )
    if div_gap > 3:
        feedback["improvement_areas"].append(
            f"Diversity: Use more varied reasoning patterns (+{div_gap:.1f} points needed)"
        )
        feedback["specific_recommendations"].append(
            "Study different diagnostic frameworks (anatomical, physiological, pathological)"
        )
    if conf_gap > 3:
        feedback["improvement_areas"].append(
            f"Confidence: Improve precision in medical terminology (+{conf_gap:.1f} points needed)"
        )
        feedback["specific_recommendations"].append(
            "Review medical terminology and practice using specific clinical language"
        )
    if dens_gap > 2:
        feedback["improvement_areas"].append(
            f"Density: Develop deeper reasoning for each medical finding (+{dens_gap:.1f} points needed)"
        )
        feedback["specific_recommendations"].append(
            "For each symptom/sign, ask: What could cause this? What tests confirm? What's the mechanism?"
        )

    # Expert examples to study
    for q in expert_questions[:3]:
        feedback["expert_examples"].append(f"Expert question: {q}")
    for h in expert_hypotheses[:3]:
        feedback["expert_examples"].append(f"Expert hypothesis: {h}")

    # Learning pathway recommendation
    if score_ratio < 0.4:
        feedback["learning_pathway"] = {
            "phase": "Phase 1: Foundation Building",
            "duration": "2-4 weeks",
            "focus": [
                "Master systematic history-taking frameworks (e.g., OPQRST for pain)",
                "Practice identifying key clinical concepts in case studies",
                "Build vocabulary of common differential diagnoses",
            ],
        }
    elif score_ratio < 0.7:
        feedback["learning_pathway"] = {
            "phase": "Phase 2: Skill Development",
            "duration": "4-6 weeks",
            "focus": [
                "Practice generating differential diagnoses under time pressure",
                "Work on integrating multiple data sources (history, exam, labs)",
                "Develop pattern recognition for common presentations",
            ],
        }
    else:
        feedback["learning_pathway"] = {
            "phase": "Phase 3: Advanced Integration",
            "duration": "4-6 weeks",
            "focus": [
                "Practice complex multi-system cases",
                "Develop expertise in specific clinical domains",
                "Focus on efficiency and clinical judgment refinement",
            ],
        }

    return feedback


def generate_adaptive_insights_with_llm(
    student_id: str,
    diagnostic_feedback: Dict,
    communication_feedback: Dict,
    client: Optional[LLMClient] = None,
    use_llm_fallback: bool = False
) -> Dict:
    """
    Generate adaptive insights using LLM for intelligent recommendations.
    """
    if not _LLM_AVAILABLE:
        if use_llm_fallback:
            return generate_adaptive_insights(student_id, diagnostic_feedback, communication_feedback)
        raise ImportError("LLM integration not available.")

    if client is None:
        client = LLMClient()

    try:
        base_insights = generate_adaptive_insights(student_id, diagnostic_feedback, communication_feedback)

        gaps = diagnostic_feedback.get("missing", [])
        performance = {
            "open_question_ratio": communication_feedback.get("open_ratio", 0.0),
            "empathy_count": communication_feedback.get("empathy_count", 0),
            "diagnostic_coverage": diagnostic_feedback.get("coverage", 0.0),
        }

        llm_recommendation = client.generate_case_recommendation(
            student_gaps=gaps,
            student_id=student_id,
            recent_performance=performance
        )

        return {
            "insights": base_insights["insights"],
            "recommended_next_case": base_insights["recommended_next_case"],
            "llm_recommendation": llm_recommendation
        }
    except Exception as e:
        if use_llm_fallback:
            return generate_adaptive_insights(student_id, diagnostic_feedback, communication_feedback)
        raise

