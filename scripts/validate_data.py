#!/usr/bin/env python3
"""
validate_data.py — Validates the CLARA pipeline against real data.

Tests:
  1. Acoustic feature extraction on cv_audio .ogg files (librosa pipeline)
  2. MIMIC-IV CSV — NLP + quality scoring + expert baseline comparison
  3. End-to-end — real audio paralinguistics wired through full CLARA pipeline

Usage:
    python scripts/validate_data.py

Expected data files:
    data/cv_audio/cv_audio/**/*.ogg   (85 VietMed audio clips)
    data/mimic_iv_summarization_test_dataset_shortened.csv  (999 MIMIC-IV notes)
"""
import sys
import csv
import glob
import statistics
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clara.asr import _extract_audio_features, _acoustic_to_paralinguistic, transcribe
from clara.nlp import analyze_utterance, extract_reasoning_patterns
from clara.semantic import align_concepts_to_kb
from clara.feedback import communication_feedback, diagnostic_feedback
from clara.analytics import (
    compute_score,
    compute_reasoning_quality_score,
    compare_to_expert_baseline,
    EXPERT_BASELINES,
)
from clara.adaptive import generate_personalized_feedback

PASS = 0
FAIL = 0

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OGG_GLOB = str(DATA_DIR / "cv_audio" / "cv_audio" / "**" / "*.ogg")
CSV_PATH = DATA_DIR / "mimic_iv_summarization_test_dataset_shortened.csv"


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {label}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  ❌ {label}" + (f"  [{detail}]" if detail else ""))


# ── Part 1: Acoustic feature extraction on cv_audio .ogg files ───────────────

def validate_audio():
    print("\n═══ Part 1: Acoustic Features (cv_audio .ogg files) ═══")

    ogg_files = sorted(glob.glob(OGG_GLOB, recursive=True))
    print(f"  ℹ️  Found {len(ogg_files)} .ogg files")
    check("85 ogg files found", len(ogg_files) == 85)

    all_feats = []
    errors = []
    for ogg in ogg_files:
        try:
            all_feats.append(_extract_audio_features(ogg))
        except Exception as e:
            errors.append((Path(ogg).name, str(e)))

    check(
        f"extraction ran on all {len(ogg_files)} files",
        len(errors) == 0,
        f"{len(all_feats)} ok, {len(errors)} errors",
    )

    f0 = all_feats[0]
    para0 = _acoustic_to_paralinguistic(f0)
    print(f"\n  Sample ({Path(ogg_files[0]).name}):")
    print(f"    avg_pitch_hz      : {f0['avg_pitch']:.2f}")
    print(f"    speaking_rate_bpm : {f0['speaking_rate']:.2f}")
    print(f"    pause_duration_sec: {f0['pause_duration']:.3f}")
    print(f"    jitter            : {f0['jitter']:.4f}" if f0['jitter'] else "    jitter            : None")
    print(f"    → emphasis={para0['emphasis']}, pacing={para0['pacing']}, pitch={para0['pitch']}")

    check("avg_pitch numeric",       isinstance(f0['avg_pitch'],     float))
    check("speaking_rate numeric",   isinstance(f0['speaking_rate'], float))
    check("pause_duration numeric",  isinstance(f0['pause_duration'],float))
    check("pitch label valid",    para0['pitch']    in ('low', 'normal', 'high'))
    check("pacing label valid",   para0['pacing']   in ('slow', 'normal', 'fast'))
    check("emphasis label valid", para0['emphasis'] in ('normal', 'high'))

    pitches = [r['avg_pitch']     for r in all_feats if r['avg_pitch'] > 0]
    tempos  = [r['speaking_rate'] for r in all_feats]
    jitters = [r['jitter']        for r in all_feats if r['jitter'] is not None]
    print(f"\n  Summary across {len(all_feats)} files:")
    print(f"    pitch  — mean={statistics.mean(pitches):.1f} Hz,  min={min(pitches):.1f},  max={max(pitches):.1f}")
    print(f"    tempo  — mean={statistics.mean(tempos):.1f} BPM, min={min(tempos):.1f}, max={max(tempos):.1f}")
    if jitters:
        print(f"    jitter — mean={statistics.mean(jitters):.4f}, min={min(jitters):.4f}, max={max(jitters):.4f}")

    return all_feats


# ── Part 2: MIMIC-IV CSV → NLP + scoring ─────────────────────────────────────

def validate_mimic():
    print("\n═══ Part 2: MIMIC-IV CSV — NLP + Scoring Pipeline ═══")

    if not CSV_PATH.exists():
        print(f"  ⚠️  CSV not found at {CSV_PATH} — skipping Part 2")
        return

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    print(f"  ℹ️  Loaded {len(rows)} MIMIC-IV notes")
    check("999 rows loaded", len(rows) == 999)

    # NLP on first 10 notes
    nlp_out = []
    for row in rows[:10]:
        text = row['text'][:800]
        nlp_out.append((analyze_utterance(text), extract_reasoning_patterns(text)))

    check("analyze_utterance on 10 notes",      len(nlp_out) == 10)
    check("concepts is list",                   isinstance(nlp_out[0][0].get('concepts'), list))
    check("reasoning_patterns has questions",   'questions'  in nlp_out[0][1])
    check("reasoning_patterns has hypotheses",  'hypotheses' in nlp_out[0][1])

    a0, p0 = nlp_out[0]
    print(f"\n  Sample note (first):")
    print(f"    concepts found    : {a0.get('concepts', [])}")
    print(f"    questions found   : {len(p0.get('questions', []))}")
    print(f"    hypotheses found  : {len(p0.get('hypotheses', []))}")

    # Quality scoring on 20 notes
    print(f"\n  Running quality scoring on 20 notes...")
    scores = []
    for row in rows[:20]:
        text = row['text'][:1000]
        p = extract_reasoning_patterns(text)
        s = compute_reasoning_quality_score(
            questions=p.get('questions', []),
            hypotheses=p.get('hypotheses', []),
            total_entities=len(analyze_utterance(text).get('concepts', [])),
            confidence_scores=[],
        )
        scores.append(s)

    check("quality scores computed for 20 notes", len(scores) == 20)
    check("all have overall_score", all('overall_score' in s for s in scores))
    check("all have grade",         all('grade' in s for s in scores))
    check("scores in 0-100 range",  all(0 <= s['overall_score'] <= 100 for s in scores))

    vals = [s['overall_score'] for s in scores]
    print(f"\n  Scoring summary (20 notes):")
    print(f"    range      : {min(vals):.1f} – {max(vals):.1f}")
    print(f"    mean       : {statistics.mean(vals):.1f}")
    print(f"    grade dist : {dict(Counter(s['grade'] for s in scores))}")

    # Expert baseline comparison on 5 notes
    print(f"\n  Expert baseline comparison on 5 notes...")
    comps = []
    for row in rows[:5]:
        text = row['text'][:1000]
        p = extract_reasoning_patterns(text)
        c = compare_to_expert_baseline(
            trainee_questions=len(p.get('questions', [])),
            trainee_hypotheses=len(p.get('hypotheses', [])),
            trainee_entities=len(analyze_utterance(text).get('concepts', [])),
            trainee_confidence=0.0,
        )
        comps.append(c)

    check("expert comparison ran on 5 notes",  len(comps) == 5)
    check("has expert key",                    all('expert'           in c for c in comps))
    check("has trainee key",                   all('trainee'          in c for c in comps))
    check("has expert_advantage key",          all('expert_advantage' in c for c in comps))

    print(f"\n  Sample (note 1):")
    print(f"    trainee questions : {comps[0]['trainee'].get('total_questions', 0)}")
    print(f"    expert questions  : {comps[0]['expert'].get('total_questions', 0)}")
    print(f"    expert advantage  : {comps[0]['expert_advantage'].get('overall', 'N/A'):.2f}x")


# ── Part 3: End-to-end — real audio paralinguistics through full pipeline ─────

def validate_end_to_end(all_feats):
    print("\n═══ Part 3: End-to-End — Audio Acoustics → CLARA Pipeline ═══")

    sample_text = (
        "Do you have chest pain? When did it start? "
        "Does it radiate to your left arm? Any shortness of breath? "
        "Do you have nausea or sweating? Any family history of heart disease?"
    )
    asr_out   = transcribe(sample_text)
    real_para = _acoustic_to_paralinguistic(all_feats[0])
    asr_out['paralinguistic'] = real_para

    analysis = analyze_utterance(asr_out['text'])
    concepts = analysis.get('concepts', [])
    semantic = align_concepts_to_kb(concepts)
    comm_fb  = communication_feedback([analysis])
    diag_fb  = diagnostic_feedback(semantic)
    score    = compute_score(diag_fb, comm_fb)
    patterns = extract_reasoning_patterns(asr_out['text'])

    trainee_score = compute_reasoning_quality_score(
        questions=patterns.get('questions', []),
        hypotheses=patterns.get('hypotheses', []),
        total_entities=len(concepts),
        confidence_scores=[],
    )
    expert_score = compute_reasoning_quality_score(
        questions=EXPERT_BASELINES.get('expert_questions', []),
        hypotheses=EXPERT_BASELINES.get('expert_hypotheses', []),
        total_entities=33,
        confidence_scores=[EXPERT_BASELINES.get('ner_confidence_mean', 0.9151)],
    )
    personalized = generate_personalized_feedback(
        trainee_score=trainee_score,
        expert_score=expert_score,
        trainee_questions=patterns.get('questions', []),
        trainee_hypotheses=patterns.get('hypotheses', []),
        expert_questions=EXPERT_BASELINES.get('expert_questions', []),
        expert_hypotheses=EXPERT_BASELINES.get('expert_hypotheses', []),
    )

    check("end-to-end pipeline completes",   True)
    check("concepts extracted",              len(concepts) > 0, str(concepts))
    check("diagnostic coverage > 0",        diag_fb.get('coverage', 0) > 0)
    check("score computed",                  'score' in score)
    check("trainee quality score computed",  'overall_score' in trainee_score)
    check("personalized feedback generated", 'overall_assessment' in personalized)
    check("real pitch wired in",             real_para['avg_pitch_hz'] is not None)

    print(f"\n  End-to-end results:")
    print(f"    concepts         : {concepts}")
    print(f"    coverage         : {diag_fb.get('coverage', 0):.1%}")
    print(f"    score            : {score.get('score')}")
    print(f"    quality          : {trainee_score['overall_score']:.1f}/100 (Grade: {trainee_score['grade']})")
    print(f"    real pitch       : {real_para['avg_pitch_hz']:.1f} Hz → {real_para['pitch']}")
    print(f"    real tempo       : {real_para['speaking_rate_bpm']:.1f} BPM → {real_para['pacing']}")
    print(f"    jitter           : {real_para['jitter']:.4f}" if real_para['jitter'] else "    jitter           : None")
    print(f"    assessment       : {personalized.get('overall_assessment', '')[:90]}")
    print(f"    strengths        : {personalized.get('strengths', [])[:2]}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  CLARA Validation — cv_audio + MIMIC-IV CSV         ║")
    print("╚══════════════════════════════════════════════════════╝")

    all_feats = validate_audio()
    validate_mimic()
    validate_end_to_end(all_feats)

    print(f"\n{'═' * 54}")
    print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    if FAIL == 0:
        print("  🎉 All validation checks passed!")
    else:
        print(f"  ⚠️  {FAIL} check(s) failed.")
    print(f"{'═' * 54}\n")

    sys.exit(0 if FAIL == 0 else 1)
