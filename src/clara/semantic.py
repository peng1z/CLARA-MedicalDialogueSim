"""Semantic alignment engine: compares asked concepts to clinical knowledge DB and finds omissions.

Also provides access to expert reasoning patterns from MIMIC-IV baselines.
"""
from typing import Dict, List
import json
from pathlib import Path

# semantic.py is located at <repo>/src/clara/semantic.py
# The data/ folder lives at the repo root. Move up two parents to reach repo root.
_KB_PATH = Path(__file__).resolve().parents[2] / "data" / "clinical_knowledge.json"


def load_kb() -> Dict:
    with open(_KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_expert_patterns(scenario: str = "scenario_chest_pain") -> Dict:
    """Load expert reasoning patterns from the knowledge base (from VietMed.ipynb MIMIC-IV analysis)."""
    kb = load_kb()
    sc = kb.get(scenario, {})
    return sc.get("expert_reasoning_patterns", {})


def align_concepts_to_kb(asked_concepts: List[str], scenario: str = "scenario_chest_pain") -> Dict:
    kb = load_kb()
    sc = kb.get(scenario, {})
    required = sc.get("required_concepts", [])
    recommended_order = sc.get("recommended_order", [])

    matched = [c for c in required if c in asked_concepts]
    missing = [c for c in required if c not in asked_concepts]

    # identify omissions that are high-priority (e.g., cardiac red flags)
    critical = [c for c in ["chest_pain", "onset_timing", "quality", "radiation", "shortness_of_breath"] if c in missing]

    # Evaluate structural sequence issues: which recommended items were asked out of order
    order_issues = []
    asked_positions = {c: idx for idx, c in enumerate(asked_concepts)}
    last_pos = -1
    for c in recommended_order:
        if c in asked_positions:
            pos = asked_positions[c]
            if pos < last_pos:
                order_issues.append(c)
            last_pos = pos

    return {
        "matched": matched,
        "missing": missing,
        "critical_missing": critical,
        "order_issues": order_issues,
        "required_count": len(required),
    }
