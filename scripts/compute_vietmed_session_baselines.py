#!/usr/bin/env python3
"""
compute_vietmed_session_baselines.py — Session-level VietMed expert advantage.

Re-aggregates per-clip transcripts (already cached) into full consultation
sessions (VietMed_007 … VietMed_011), then re-extracts reasoning patterns
using forced dialogue mode to get the correct session-level ratio.

Compares against:
  - VIETMED_BASELINES  (expert clinician per-session: 33 q / 85 h)
  - MIMIC_BASELINES    (prose expert per-note:  3.45 q / 1.04 h)
"""
import sys
import json
import statistics
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clara.nlp import (
    _QUESTION_PATTERNS,
    _HYPOTHESIS_PATTERNS,
    extract_concepts,
)
from clara.analytics import (
    compute_reasoning_quality_score,
    MIMIC_BASELINES,
    VIETMED_BASELINES,
)

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "vietmed_transcripts_cache.json"
OUT_PATH   = Path(__file__).resolve().parents[1] / "data" / "vietmed_session_baselines.json"


def extract_dialogue_patterns(text: str):
    """Force dialogue mode: apply conversational question/hypothesis patterns."""
    questions, hypotheses = [], []
    for pat in _QUESTION_PATTERNS:
        questions.extend(pat.findall(text))
    for pat in _HYPOTHESIS_PATTERNS:
        hypotheses.extend(pat.findall(text))
    return questions, hypotheses


def grade(score: float) -> str:
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    return "F"


def main():
    if not CACHE_PATH.exists():
        print(f"❌ Cache not found: {CACHE_PATH}")
        print("   Run compute_vietmed_baselines.py first to build the transcript cache.")
        sys.exit(1)

    with open(CACHE_PATH) as f:
        cache = json.load(f)

    # Group clips by session directory
    sessions: dict = defaultdict(list)
    for path, data in cache.items():
        session = Path(path).parent.name
        sessions[session].append((path, data))

    print(f"Found {len(sessions)} VietMed sessions ({len(cache)} total clips)\n")

    all_q, all_h, all_scores, all_c = [], [], [], []

    for session in sorted(sessions.keys()):
        clips = sessions[session]
        # Sort clips by start-time embedded in filename
        clips.sort(key=lambda x: float(Path(x[0]).stem.split("_")[0]))
        full_text = " ".join(d["en"] for _, d in clips if d.get("en"))

        q_list, h_list = extract_dialogue_patterns(full_text)
        concepts = extract_concepts(full_text)
        q, h, c = len(q_list), len(h_list), len(concepts)

        score = compute_reasoning_quality_score(
            questions=q_list,
            hypotheses=h_list,
            total_entities=c,
            confidence_scores=[],
        )

        all_q.append(q)
        all_h.append(h)
        all_c.append(c)
        all_scores.append(score["overall_score"])

        print(f"  {session} ({len(clips):2d} clips) | q={q:3d}  h={h:3d}  c={c:3d}  "
              f"score={score['overall_score']:5.1f}  [{score['grade']}]")

    print()

    avg_q     = statistics.mean(all_q)
    avg_h     = statistics.mean(all_h)
    avg_c     = statistics.mean(all_c)
    avg_score = statistics.mean(all_scores)
    sd_score  = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

    trainee_total = avg_q + avg_h

    # ── Expert advantage vs VIETMED_BASELINES (expert clinician sessions) ─────
    exp_q_viet = VIETMED_BASELINES["questions"]   # 33
    exp_h_viet = VIETMED_BASELINES["hypotheses"]  # 85
    expert_total_viet = exp_q_viet + exp_h_viet
    ratio_viet = expert_total_viet / max(trainee_total, 0.01)

    # ── Expert advantage vs MIMIC_BASELINES (prose notes) ────────────────────
    exp_q_mimic = MIMIC_BASELINES["questions"]    # 3.45
    exp_h_mimic = MIMIC_BASELINES["hypotheses"]   # 1.04
    expert_total_mimic = exp_q_mimic + exp_h_mimic
    ratio_mimic = expert_total_mimic / max(trainee_total, 0.01)

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"{'═' * 60}")
    print(f"  VietMed SESSION-LEVEL Trainee Baselines  (n={len(all_scores)} sessions)")
    print(f"{'═' * 60}")

    print(f"\n  Reasoning Elements (per session):")
    print(f"    avg questions  : {avg_q:.2f}  (total={sum(all_q)})")
    print(f"    avg hypotheses : {avg_h:.2f}  (total={sum(all_h)})")
    print(f"    avg concepts   : {avg_c:.2f}")

    print(f"\n  Quality Score:")
    print(f"    mean  : {avg_score:.2f}")
    print(f"    stdev : {sd_score:.2f}")
    print(f"    grades: {dict(Counter(grade(s) for s in all_scores))}")

    print(f"\n  Expert vs Trainee (VIETMED expert sessions):")
    print(f"    expert  : q={exp_q_viet}  h={exp_h_viet}  total={expert_total_viet}")
    print(f"    trainee : q={avg_q:.2f}  h={avg_h:.2f}  total={trainee_total:.2f}")
    print(f"    ⭐ Expert advantage : {ratio_viet:.2f}x")

    print(f"\n  Expert vs Trainee (MIMIC prose notes):")
    print(f"    expert  : q={exp_q_mimic}  h={exp_h_mimic}  total={expert_total_mimic:.2f}")
    print(f"    trainee : q={avg_q:.2f}  h={avg_h:.2f}  total={trainee_total:.2f}")
    print(f"    ⭐ Expert advantage : {ratio_mimic:.2f}x")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "_source": "vietmed_transcripts_cache.json (session-aggregated, dialogue mode)",
        "total_sessions": len(all_scores),
        "total_clips": len(cache),
        "per_session": {
            "avg_questions":  round(avg_q, 2),
            "avg_hypotheses": round(avg_h, 2),
            "avg_concepts":   round(avg_c, 2),
        },
        "score_distribution": {
            "mean":  round(avg_score, 2),
            "stdev": round(sd_score, 2),
            "min":   round(min(all_scores), 2),
            "max":   round(max(all_scores), 2),
        },
        "expert_vs_trainee": {
            "vietmed_expert": {
                "questions": exp_q_viet,
                "hypotheses": exp_h_viet,
                "total": expert_total_viet,
                "advantage_x": round(ratio_viet, 2),
            },
            "mimic_expert": {
                "questions": exp_q_mimic,
                "hypotheses": exp_h_mimic,
                "total": round(expert_total_mimic, 2),
                "advantage_x": round(ratio_mimic, 2),
            },
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✅ Saved to {OUT_PATH}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
