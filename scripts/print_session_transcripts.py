#!/usr/bin/env python3
"""
print_session_transcripts.py — Print Vietnamese + English transcripts for one session.

Usage:
    python scripts/print_session_transcripts.py [SESSION_NAME]

    SESSION_NAME defaults to VietMed_007 (smallest session, good for validation).
    Available: VietMed_007, VietMed_008, VietMed_009, VietMed_010, VietMed_011
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "vietmed_transcripts_cache.json"

SESSION = sys.argv[1] if len(sys.argv) > 1 else "VietMed_007"


def main():
    if not CACHE_PATH.exists():
        print(f"❌ Cache not found: {CACHE_PATH}")
        print("   Run compute_vietmed_baselines.py first.")
        sys.exit(1)

    with open(CACHE_PATH) as f:
        cache = json.load(f)

    sessions = defaultdict(list)
    for path, data in cache.items():
        session_name = Path(path).parent.name
        sessions[session_name].append((Path(path).name, data))

    if SESSION not in sessions:
        print(f"❌ Session '{SESSION}' not found. Available: {sorted(sessions.keys())}")
        sys.exit(1)

    clips = sorted(sessions[SESSION], key=lambda x: float(x[0].split("_")[0]))

    print(f"{'═' * 60}")
    print(f"  Session: {SESSION}  ({len(clips)} clips)")
    print(f"{'═' * 60}\n")

    for fname, data in clips:
        start = fname.split("_")[0]
        end   = fname.split("_")[1].replace(".ogg", "")
        print(f"[{start}s – {end}s]  ({fname})")
        print(f"  VI: {data.get('vi', '(empty)')}")
        print(f"  EN: {data.get('en', '(empty)')}")
        print()

    # Concatenated full-session English text
    full_en = " ".join(d.get("en", "") for _, d in clips)
    print(f"{'─' * 60}")
    print(f"  Full session (English, concatenated):\n")
    # Word-wrap at ~80 chars
    words = full_en.split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > 80:
            print(f"  {line}")
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        print(f"  {line}")
    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()
