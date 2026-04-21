#!/usr/bin/env python3
"""
test_integration.py — Validates all 4 Tier 1 integration tasks.

Tests:
  1. NER model integration in nlp.py (extract_medical_entities, concept extraction)
  2. Reasoning quality scoring in analytics.py (0-100 scale, A-F grades)
  3. Expert baselines in clinical_knowledge.json (MIMIC-IV patterns loaded)
  4. Personalized feedback in adaptive.py (strengths, gaps, learning pathways)

Run from repo root:
    python scripts/test_integration.py
"""
import sys
import os

# Add src/ to path so we can import clara package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {label}")
    else:
        FAIL += 1
        print(f"  ❌ {label}" + (f" — {detail}" if detail else ""))


def test_task1_ner_integration():
    """Task 1: Medical NER model plugged into nlp.py"""
    print("\n═══ Task 1: NER Model Integration (nlp.py) ═══")

    from clara.nlp import (
        extract_medical_entities,
        extract_concepts,
        analyze_utterance,
        extract_reasoning_patterns,
        _HF_NER_AVAILABLE,
    )

    # Check if HuggingFace transformers is importable
    print(f"  ℹ️  HuggingFace NER available: {_HF_NER_AVAILABLE}")

    # --- extract_medical_entities ---
    if _HF_NER_AVAILABLE:
        entities = extract_medical_entities("Patient presents with chest pain and shortness of breath.")
        check("extract_medical_entities returns list", isinstance(entities, list))
        check("extract_medical_entities finds entities", len(entities) > 0,
              f"got {len(entities)} entities")
        if entities:
            e = entities[0]
            check("entity has 'word' key", "word" in e)
            check("entity has 'entity_group' key", "entity_group" in e)
            check("entity has 'score' key", "score" in e)
            check("entity score is float 0-1", 0 <= e["score"] <= 1, f"score={e['score']}")
    else:
        print("  ⚠️  Skipping NER model tests (transformers not installed)")
        print("     Install with: pip install transformers torch")

    # --- extract_concepts (should work regardless of NER availability) ---
    concepts = extract_concepts("I have chest pain radiating to my left arm with shortness of breath")
    check("extract_concepts returns list", isinstance(concepts, list))
    check("extract_concepts finds chest_pain", "chest_pain" in concepts, f"got {concepts}")
    check("extract_concepts finds radiation", "radiation" in concepts, f"got {concepts}")
    check("extract_concepts finds shortness_of_breath", "shortness_of_breath" in concepts, f"got {concepts}")

    # --- analyze_utterance now includes entities + reasoning ---
    result = analyze_utterance("Do you have any chest pain? I think it could be cardiac.")
    check("analyze_utterance has 'medical_entities' key", "medical_entities" in result)
    check("analyze_utterance has 'reasoning_patterns' key", "reasoning_patterns" in result)
    check("analyze_utterance type is open/closed", result["type"] in ["open", "closed", "unknown"])

    # --- extract_reasoning_patterns ---
    text = "Do you have chest pain? I think it's likely acute coronary syndrome. Rule out PE."
    patterns = extract_reasoning_patterns(text)
    check("reasoning_patterns has 'questions' key", "questions" in patterns)
    check("reasoning_patterns has 'hypotheses' key", "hypotheses" in patterns)
    check("reasoning_patterns finds questions", len(patterns["questions"]) > 0,
          f"got {len(patterns['questions'])}")
    check("reasoning_patterns finds hypotheses", len(patterns["hypotheses"]) > 0,
          f"got {len(patterns['hypotheses'])}")


def test_task2_scoring_framework():
    """Task 2: VietMed scoring framework in analytics.py"""
    print("\n═══ Task 2: Reasoning Quality Scoring (analytics.py) ═══")

    from clara.analytics import (
        compute_score,
        compute_reasoning_quality_score,
        compare_to_expert_baseline,
        EXPERT_BASELINES,
    )

    # --- EXPERT_BASELINES loaded ---
    check("EXPERT_BASELINES is dict", isinstance(EXPERT_BASELINES, dict))
    check("EXPERT_BASELINES has questions", "questions" in EXPERT_BASELINES,
          f"keys: {list(EXPERT_BASELINES.keys())}")
    check("EXPERT_BASELINES questions == 33", EXPERT_BASELINES.get("questions") == 33)
    check("EXPERT_BASELINES hypotheses == 85", EXPERT_BASELINES.get("hypotheses") == 85)

    # --- compute_reasoning_quality_score ---
    score = compute_reasoning_quality_score(
        questions=["q1", "q2", "q3"],
        hypotheses=["h1", "h2"],
        total_entities=10,
        confidence_scores=[0.9, 0.85, 0.92],
    )
    check("score has overall_score", "overall_score" in score)
    check("score has grade", "grade" in score)
    check("overall_score is 0-100", 0 <= score["overall_score"] <= 100,
          f"got {score['overall_score']}")
    check("grade is A-F", score["grade"] in ["A", "B", "C", "D", "F"],
          f"got {score['grade']}")
    check("has volume_score", "volume_score" in score)
    check("has diversity_score", "diversity_score" in score)
    check("has confidence_score", "confidence_score" in score)
    check("has density_score", "density_score" in score)

    # --- compare_to_expert_baseline ---
    comparison = compare_to_expert_baseline(
        trainee_questions=5,
        trainee_hypotheses=10,
        trainee_entities=20,
        trainee_confidence=0.85,
    )
    check("comparison has 'expert' key", "expert" in comparison)
    check("comparison has 'trainee' key", "trainee" in comparison)
    check("comparison has 'expert_advantage' key", "expert_advantage" in comparison)
    adv = comparison["expert_advantage"]
    check("expert_advantage overall > 1", adv["overall"] > 1,
          f"got {adv['overall']}x")

    # --- original compute_score still works ---
    old_score = compute_score({"coverage": 0.8}, {"open_ratio": 0.6})
    check("compute_score still works", "score" in old_score, f"got {old_score}")


def test_task3_expert_baselines():
    """Task 3: Expert reasoning patterns in clinical_knowledge.json"""
    print("\n═══ Task 3: Expert Baselines in Knowledge DB ═══")

    from clara.semantic import load_kb, get_expert_patterns

    kb = load_kb()
    sc = kb.get("scenario_chest_pain", {})

    # --- Original fields still present ---
    check("required_concepts exists", "required_concepts" in sc)
    check("recommended_order exists", "recommended_order" in sc)
    check("12 required concepts", len(sc.get("required_concepts", [])) == 12)

    # --- New expert_reasoning_patterns ---
    check("expert_reasoning_patterns exists", "expert_reasoning_patterns" in sc)

    patterns = sc.get("expert_reasoning_patterns", {})
    check("expert_questions exists", "expert_questions" in patterns)
    check("expert_hypotheses exists", "expert_hypotheses" in patterns)
    check("expert_metrics exists", "expert_metrics" in patterns)

    eq = patterns.get("expert_questions", [])
    eh = patterns.get("expert_hypotheses", [])
    check(f"expert_questions has {len(eq)} items", len(eq) >= 30, f"got {len(eq)}")
    check(f"expert_hypotheses has {len(eh)} items", len(eh) >= 25, f"got {len(eh)}")

    metrics = patterns.get("expert_metrics", {})
    check("metrics has total_questions", "total_questions" in metrics)
    check("metrics has ner_confidence_mean", "ner_confidence_mean" in metrics)
    check("metrics confidence ~0.915", abs(metrics.get("ner_confidence_mean", 0) - 0.9151) < 0.01)

    # --- get_expert_patterns helper ---
    ep = get_expert_patterns()
    check("get_expert_patterns returns dict", isinstance(ep, dict))
    check("get_expert_patterns has expert_questions", "expert_questions" in ep)


def test_task4_personalized_feedback():
    """Task 4: Personalized feedback generator in adaptive.py"""
    print("\n═══ Task 4: Personalized Feedback (adaptive.py) ═══")

    from clara.adaptive import generate_personalized_feedback, generate_adaptive_insights

    # Simulate expert score (high performer)
    expert_score = {
        "volume_score": 30.0,
        "diversity_score": 25.0,
        "confidence_score": 22.9,
        "density_score": 0.3,
        "overall_score": 78.1,
        "grade": "B",
    }

    # Simulate trainee score (lower performer)
    trainee_score = {
        "volume_score": 10.0,
        "diversity_score": 12.0,
        "confidence_score": 18.0,
        "density_score": 3.0,
        "overall_score": 43.0,
        "grade": "C",
    }

    feedback = generate_personalized_feedback(
        trainee_score=trainee_score,
        expert_score=expert_score,
        trainee_questions=["Do you have chest pain?", "Any shortness of breath?"],
        trainee_hypotheses=["Could be ACS."],
    )

    check("feedback has overall_assessment", bool(feedback.get("overall_assessment")))
    check("feedback has strengths list", isinstance(feedback.get("strengths"), list))
    check("feedback has improvement_areas", isinstance(feedback.get("improvement_areas"), list))
    check("feedback has specific_recommendations", isinstance(feedback.get("specific_recommendations"), list))
    check("feedback has expert_examples", isinstance(feedback.get("expert_examples"), list))
    check("feedback has learning_pathway", isinstance(feedback.get("learning_pathway"), dict))

    # With this trainee score (43/78.1 = 55%), should be Phase 2
    pathway = feedback.get("learning_pathway", {})
    check("learning_pathway has phase", "phase" in pathway)
    check("learning_pathway has focus items", len(pathway.get("focus", [])) > 0)

    # Improvement areas should be populated (trainee is weaker)
    check("improvement_areas not empty", len(feedback["improvement_areas"]) > 0,
          f"got {len(feedback['improvement_areas'])} areas")

    # Expert examples loaded from KB
    check("expert_examples not empty", len(feedback["expert_examples"]) > 0,
          f"got {len(feedback['expert_examples'])} examples")

    # --- original generate_adaptive_insights still works ---
    old = generate_adaptive_insights("test_student", {"missing": ["radiation"]}, {"open_ratio": 0.5})
    check("generate_adaptive_insights still works", "insights" in old)


def test_end_to_end():
    """End-to-end: simulate a trainee session using all integrated components."""
    print("\n═══ End-to-End Integration Test ═══")

    from clara.nlp import analyze_utterance, extract_reasoning_patterns
    from clara.semantic import align_concepts_to_kb, get_expert_patterns
    from clara.feedback import communication_feedback, diagnostic_feedback
    from clara.analytics import compute_score, compute_reasoning_quality_score, compare_to_expert_baseline
    from clara.adaptive import generate_personalized_feedback

    # Simulate a trainee asking questions
    utterances = [
        "Tell me about your chest pain. When did it start?",
        "Is the pain sharp or dull? How severe is it?",
        "Does it radiate anywhere?",
        "Are you short of breath?",
        "I'm sorry to hear that. Have you been sweating?",
        "Any nausea or vomiting?",
        "Do you have any history of heart disease?",
        "What medications are you taking?",
    ]

    # Run each utterance through the pipeline
    analyses = []
    all_concepts = []
    all_entities = []
    all_confidence = []

    for utt in utterances:
        result = analyze_utterance(utt)
        analyses.append(result)
        all_concepts.extend(result["concepts"])
        for ent in result.get("medical_entities", []):
            all_entities.append(ent)
            all_confidence.append(ent.get("score", 0))

    unique_concepts = sorted(set(all_concepts))
    print(f"  ℹ️  Concepts extracted: {unique_concepts}")
    print(f"  ℹ️  NER entities found: {len(all_entities)}")

    # Semantic alignment
    alignment = align_concepts_to_kb(unique_concepts)
    check("alignment has matched", len(alignment["matched"]) > 0,
          f"matched {len(alignment['matched'])}/{alignment['required_count']}")
    check("alignment has missing", isinstance(alignment["missing"], list))

    # Feedback
    comm_fb = communication_feedback(analyses)
    diag_fb = diagnostic_feedback(alignment)
    check("communication feedback generated", bool(comm_fb.get("notes")))
    check("diagnostic feedback generated", "coverage" in diag_fb)
    print(f"  ℹ️  Coverage: {diag_fb['coverage']:.0%}")
    print(f"  ℹ️  Missing: {diag_fb['missing']}")

    # Scoring
    old_score = compute_score(diag_fb, comm_fb)
    check("old score computed", "score" in old_score, f"score={old_score['score']}")

    # Reasoning patterns from all utterances
    full_text = " ".join(utterances)
    patterns = extract_reasoning_patterns(full_text)
    trainee_q = patterns["questions"]
    trainee_h = patterns["hypotheses"]

    # New quality score
    quality = compute_reasoning_quality_score(
        questions=trainee_q,
        hypotheses=trainee_h,
        total_entities=len(all_entities) or 1,
        confidence_scores=all_confidence or [0.8],
    )
    check("quality score computed", "overall_score" in quality)
    print(f"  ℹ️  Quality: {quality['overall_score']}/100 (Grade: {quality['grade']})")

    # Expert comparison
    comparison = compare_to_expert_baseline(
        trainee_questions=len(trainee_q),
        trainee_hypotheses=len(trainee_h),
        trainee_entities=len(all_entities),
        trainee_confidence=sum(all_confidence) / len(all_confidence) if all_confidence else 0.8,
    )
    check("expert comparison computed", "expert_advantage" in comparison)
    print(f"  ℹ️  Expert advantage: {comparison['expert_advantage']['overall']}x")

    # Expert baseline score for comparison
    expert_patterns = get_expert_patterns()
    expert_quality = compute_reasoning_quality_score(
        questions=expert_patterns.get("expert_questions", []),
        hypotheses=expert_patterns.get("expert_hypotheses", []),
        total_entities=4447,
        confidence_scores=[0.9151] * 100,
    )

    # Personalized feedback
    pfb = generate_personalized_feedback(
        trainee_score=quality,
        expert_score=expert_quality,
        trainee_questions=trainee_q,
        trainee_hypotheses=trainee_h,
    )
    check("personalized feedback generated", bool(pfb.get("overall_assessment")))
    print(f"  ℹ️  Assessment: {pfb['overall_assessment']}")
    print(f"  ℹ️  Strengths: {pfb['strengths']}")
    print(f"  ℹ️  Pathway: {pfb['learning_pathway'].get('phase', 'N/A')}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  CLARA Integration Test — Tier 1 (4 Tasks)         ║")
    print("╚══════════════════════════════════════════════════════╝")

    test_task1_ner_integration()
    test_task2_scoring_framework()
    test_task3_expert_baselines()
    test_task4_personalized_feedback()
    test_end_to_end()

    print(f"\n{'═' * 50}")
    print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    if FAIL == 0:
        print("  🎉 All Tier 1 integration tasks verified!")
    else:
        print(f"  ⚠️  {FAIL} check(s) need attention")
    print(f"{'═' * 50}")

    sys.exit(1 if FAIL > 0 else 0)
