#!/usr/bin/env python3
"""
compute_vietmed_baselines.py — Derives trainee baselines from VietMed audio.

Pipeline per .ogg file:
  1. Whisper ASR       → Vietnamese transcript
  2. deep-translator   → English translation
  3. extract_reasoning_patterns() → questions + hypotheses
  4. compute_reasoning_quality_score() → 0-100 score

Then computes expert/trainee ratio (the 2.8x claim) by comparing
against MIMIC_BASELINES from analytics.py.

Usage:
    python scripts/compute_vietmed_baselines.py

Outputs:
    - Console report with all statistics and expert/trainee ratio
    - data/vietmed_baselines.json
"""
import sys
import json
import glob
import statistics
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clara.nlp import extract_reasoning_patterns, analyze_utterance
from clara.analytics import compute_reasoning_quality_score, MIMIC_BASELINES

OGG_GLOB   = str(Path(__file__).resolve().parents[1] / "data" / "cv_audio" / "cv_audio" / "**" / "*.ogg")
OUT_PATH   = Path(__file__).resolve().parents[1] / "data" / "vietmed_baselines.json"
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "vietmed_transcripts_cache.json"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def transcribe_ogg(path: str, model) -> str:
    """Transcribe a .ogg file to text using Whisper."""
    result = model.transcribe(path, language="vi")
    return result["text"].strip()


def translate_to_english(text: str) -> str:
    """Translate Vietnamese text to English using deep-translator."""
    if not text:
        return ""
    try:
        from deep_translator import GoogleTranslator
        # deep-translator handles long text by chunking
        translated = GoogleTranslator(source="vi", target="en").translate(text)
        return translated or text
    except Exception as e:
        print(f"    ⚠️  Translation failed: {e} — using original text")
        return text


def _to_python(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


def grade(score: float) -> str:
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    return "F"


def main():
    ogg_files = sorted(glob.glob(OGG_GLOB, recursive=True))
    n = len(ogg_files)
    if n == 0:
        print(f"❌ No .ogg files found at {OGG_GLOB}")
        sys.exit(1)

    print(f"Found {n} VietMed .ogg files.")
    print("Loading Whisper base model...")

    import whisper
    model = whisper.load_model("base")
    print("✅ Whisper loaded.\n")

    cache = load_cache()
    cached_count = sum(1 for f in ogg_files if f in cache)
    print(f"Cache: {cached_count}/{n} already transcribed.\n")

    all_questions   = []
    all_hypotheses  = []
    all_scores      = []
    all_concepts    = []
    errors          = []
    translations    = {}

    for i, ogg in enumerate(ogg_files):
        fname = Path(ogg).name
        print(f"  [{i+1:02d}/{n}] {fname}", end=" ", flush=True)

        try:
            # Step 1: ASR (cached)
            if ogg in cache:
                vi_text = cache[ogg]["vi"]
                en_text = cache[ogg]["en"]
                print(f"(cached)", end=" ", flush=True)
            else:
                vi_text = transcribe_ogg(ogg, model)
                print(f"→ ASR done", end=" ", flush=True)

                # Step 2: Translate
                en_text = translate_to_english(vi_text)
                print(f"→ translated", end=" ", flush=True)

                cache[ogg] = {"vi": vi_text, "en": en_text}
                save_cache(cache)

            translations[fname] = {"vi": vi_text, "en": en_text}

            # Step 3: Extract reasoning patterns
            patterns = extract_reasoning_patterns(en_text)
            analysis = analyze_utterance(en_text)
            concepts = analysis.get("concepts", [])

            q = len(patterns.get("questions", []))
            h = len(patterns.get("hypotheses", []))
            c = len(concepts)

            # Step 4: Quality score
            score = compute_reasoning_quality_score(
                questions=patterns.get("questions", []),
                hypotheses=patterns.get("hypotheses", []),
                total_entities=c,
                confidence_scores=[],
            )

            all_questions.append(q)
            all_hypotheses.append(h)
            all_concepts.append(c)
            all_scores.append(score["overall_score"])

            print(f"→ q={q} h={h} c={c} score={score['overall_score']:.1f}")

        except Exception as e:
            errors.append((fname, str(e)))
            print(f"→ ❌ ERROR: {e}")

    print(f"\nProcessed {len(all_scores)}/{n} files. Errors: {len(errors)}")

    if not all_scores:
        print("No results — exiting.")
        sys.exit(1)

    # ── Statistics ────────────────────────────────────────────────────────────
    avg_q     = statistics.mean(all_questions)
    avg_h     = statistics.mean(all_hypotheses)
    avg_c     = statistics.mean(all_concepts)
    avg_score = statistics.mean(all_scores)
    sd_score  = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

    total_reasoning_trainee = avg_q + avg_h
    total_reasoning_expert  = MIMIC_BASELINES["questions"] + MIMIC_BASELINES["hypotheses"]
    expert_advantage        = total_reasoning_expert / max(total_reasoning_trainee, 0.01)

    print(f"\n{'═' * 56}")
    print(f"  VietMed Trainee Baselines  (n={len(all_scores)})")
    print(f"{'═' * 56}")
    print(f"\n  Reasoning Elements (per clip):")
    print(f"    avg questions  : {avg_q:.2f}  (total={sum(all_questions)})")
    print(f"    avg hypotheses : {avg_h:.2f}  (total={sum(all_hypotheses)})")
    print(f"    avg concepts   : {avg_c:.2f}  (total={sum(all_concepts)})")

    print(f"\n  Quality Score:")
    print(f"    mean  : {avg_score:.2f}")
    print(f"    stdev : {sd_score:.2f}")
    print(f"    min   : {min(all_scores):.2f}")
    print(f"    max   : {max(all_scores):.2f}")
    print(f"    grades: {dict(Counter(grade(s) for s in all_scores))}")

    print(f"\n  Expert vs Trainee Comparison (MIMIC vs VietMed):")
    print(f"    expert  avg reasoning: {total_reasoning_expert:.2f} (q={MIMIC_BASELINES['questions']}, h={MIMIC_BASELINES['hypotheses']})")
    print(f"    trainee avg reasoning: {total_reasoning_trainee:.2f} (q={avg_q:.2f}, h={avg_h:.2f})")
    print(f"    ⭐ Expert advantage   : {expert_advantage:.2f}x")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    baselines = {
        "_source": "cv_audio VietMed .ogg files, Whisper base ASR + deep-translator Vi→En",
        "total_clips": len(all_scores),
        "errors": len(errors),
        "avg_questions":  round(avg_q, 2),
        "avg_hypotheses": round(avg_h, 2),
        "avg_concepts":   round(avg_c, 2),
        "score_distribution": {
            "mean":  round(avg_score, 2),
            "stdev": round(sd_score,  2),
            "min":   round(min(all_scores), 2),
            "max":   round(max(all_scores), 2),
        },
        "expert_vs_trainee": {
            "expert_avg_reasoning":  round(total_reasoning_expert,  2),
            "trainee_avg_reasoning": round(total_reasoning_trainee, 2),
            "expert_advantage_x":    round(expert_advantage, 2),
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(_to_python(baselines), f, indent=2)
    print(f"\n  ✅ Saved to {OUT_PATH}")
    print(f"  ✅ Transcripts cached at {CACHE_PATH}")

    if errors:
        print(f"\n  ⚠️  Failed files:")
        for fname, err in errors:
            print(f"    {fname}: {err}")

    print(f"\n{'═' * 56}\n")


if __name__ == "__main__":
    main()
