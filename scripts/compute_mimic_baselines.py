#!/usr/bin/env python3
"""
compute_mimic_baselines.py — Derives expert baselines from the MIMIC-IV CSV.

Runs the full NLP + scoring pipeline across all 999 MIMIC-IV discharge notes
and computes real data-driven statistics to replace the hardcoded constants
in src/clara/analytics.py (EXPERT_BASELINES, peer_avg, peer_sd).

Usage:
    python scripts/compute_mimic_baselines.py

Outputs:
    - Console report of all statistics
    - data/mimic_baselines.json  (machine-readable, for loading at runtime)
"""
import sys
import csv
import json
import statistics
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clara.nlp import analyze_utterance, extract_reasoning_patterns
from clara.analytics import compute_reasoning_quality_score

CSV_PATH  = Path(__file__).resolve().parents[1] / "data" / "mimic_iv_summarization_test_dataset_shortened.csv"
OUT_PATH  = Path(__file__).resolve().parents[1] / "data" / "mimic_baselines.json"


def grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def main():
    if not CSV_PATH.exists():
        print(f"❌ CSV not found: {CSV_PATH}")
        sys.exit(1)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    print(f"Processing {n} MIMIC-IV notes...")

    all_scores      = []
    all_questions   = []
    all_hypotheses  = []
    all_entities    = []
    all_confidences = []
    entity_types    = Counter()

    for i, row in enumerate(rows):
        if i % 100 == 0:
            print(f"  {i}/{n}...")

        text     = row['text'][:2000]
        analysis = analyze_utterance(text)
        patterns = extract_reasoning_patterns(text)

        concepts   = analysis.get('concepts', [])
        questions  = patterns.get('questions', [])
        hypotheses = patterns.get('hypotheses', [])
        entities   = analysis.get('medical_entities', [])

        for e in entities:
            entity_types[e.get('entity_group', e.get('type', 'unknown'))] += 1

        score = compute_reasoning_quality_score(
            questions=questions,
            hypotheses=hypotheses,
            total_entities=len(concepts),
            confidence_scores=[e.get('score', 0.0) for e in entities] if entities else [],
        )

        all_scores.append(score['overall_score'])
        all_questions.append(len(questions))
        all_hypotheses.append(len(hypotheses))
        all_entities.append(len(concepts))
        if entities:
            all_confidences.extend([e.get('score', 0.0) for e in entities])

    print(f"  {n}/{n}... done.\n")

    # ── Score distribution ────────────────────────────────────────────────────
    sorted_scores = sorted(all_scores)
    score_mean = statistics.mean(all_scores)
    score_sd   = statistics.stdev(all_scores)
    p25  = sorted_scores[int(n * 0.25)]
    p50  = sorted_scores[int(n * 0.50)]
    p75  = sorted_scores[int(n * 0.75)]
    p90  = sorted_scores[int(n * 0.90)]

    grade_dist = Counter(grade(s) for s in all_scores)

    # ── Reasoning elements ────────────────────────────────────────────────────
    avg_questions  = statistics.mean(all_questions)
    avg_hypotheses = statistics.mean(all_hypotheses)
    avg_entities   = statistics.mean(all_entities)

    # ── NER confidence ────────────────────────────────────────────────────────
    if all_confidences:
        conf_mean      = statistics.mean(all_confidences)
        high_conf_pct  = sum(1 for c in all_confidences if c >= 0.9) / len(all_confidences) * 100
    else:
        conf_mean     = 0.0
        high_conf_pct = 0.0

    # ── Print report ──────────────────────────────────────────────────────────
    print("═" * 56)
    print(f"  MIMIC-IV Expert Baselines  (n={n})")
    print("═" * 56)

    print(f"\n  Score Distribution:")
    print(f"    mean   : {score_mean:.2f}")
    print(f"    stdev  : {score_sd:.2f}")
    print(f"    min    : {min(all_scores):.2f}")
    print(f"    p25    : {p25:.2f}")
    print(f"    p50    : {p50:.2f}")
    print(f"    p75    : {p75:.2f}")
    print(f"    p90    : {p90:.2f}")
    print(f"    max    : {max(all_scores):.2f}")
    print(f"    grades : {dict(grade_dist)}")

    print(f"\n  Reasoning Elements (per note):")
    print(f"    avg questions  : {avg_questions:.2f}  (total={sum(all_questions)})")
    print(f"    avg hypotheses : {avg_hypotheses:.2f}  (total={sum(all_hypotheses)})")
    print(f"    avg concepts   : {avg_entities:.2f}  (total={sum(all_entities)})")

    print(f"\n  NER Confidence:")
    if all_confidences:
        print(f"    confidence_mean      : {conf_mean:.4f}")
        print(f"    high_conf_pct (≥0.9) : {high_conf_pct:.1f}%")
    else:
        print(f"    (no NER entities — transformers not installed, using regex fallback)")

    print(f"\n  Entity Types: {dict(entity_types)}")

    # ── Build baselines dict ──────────────────────────────────────────────────
    baselines = {
        "_source": "mimic_iv_summarization_test_dataset_shortened.csv",
        "_notes": (
            "Computed by scripts/compute_mimic_baselines.py. "
            "Replace EXPERT_BASELINES in analytics.py with these values."
        ),
        "total_documents": n,
        "total_entities": sum(all_entities),
        "avg_entities_per_doc": round(avg_entities, 2),
        "questions": round(avg_questions, 2),
        "hypotheses": round(avg_hypotheses, 2),
        "confidence_mean": round(conf_mean, 4),
        "high_confidence_pct": round(high_conf_pct, 1),
        "entity_types": dict(entity_types),
        "score_distribution": {
            "mean": round(score_mean, 2),
            "stdev": round(score_sd, 2),
            "min": round(min(all_scores), 2),
            "p25": round(p25, 2),
            "p50": round(p50, 2),
            "p75": round(p75, 2),
            "p90": round(p90, 2),
            "max": round(max(all_scores), 2),
        },
        "grade_distribution": dict(grade_dist),
        # peer_avg and peer_sd for compute_score() in analytics.py
        "peer_avg": round(score_mean / 100, 4),
        "peer_sd":  round(score_sd  / 100, 4),
    }

    # ── Save JSON (convert numpy types to native Python for serialization) ────
    def _to_python(obj):
        if hasattr(obj, 'item'):   # numpy scalar (float32, int64, etc.)
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        return obj

    with open(OUT_PATH, "w") as f:
        json.dump(_to_python(baselines), f, indent=2)

    print(f"\n  ✅ Saved to {OUT_PATH}")

    # ── Print what to paste into analytics.py (use _to_python'd copy) ────────
    b = _to_python(baselines)
    print(f"\n{'═' * 56}")
    print("  Paste into src/clara/analytics.py:")
    print(f"{'═' * 56}")
    print(f"""
EXPERT_BASELINES = {{
    \"total_documents\":    {b['total_documents']},
    \"total_entities\":     {b['total_entities']},
    \"avg_entities_per_doc\": {b['avg_entities_per_doc']},
    \"questions\":          {b['questions']},
    \"hypotheses\":         {b['hypotheses']},
    \"confidence_mean\":    {b['confidence_mean']},
    \"high_confidence_pct\": {b['high_confidence_pct']},
    \"entity_types\":       {b['entity_types']},
    \"score_distribution\": {json.dumps(b['score_distribution'])},
}}

# In compute_score():
peer_avg = {b['peer_avg']}
peer_sd  = {b['peer_sd']}
""")


if __name__ == "__main__":
    main()
