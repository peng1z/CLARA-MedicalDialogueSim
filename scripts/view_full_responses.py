#!/usr/bin/env python3
"""View full LLM responses side-by-side for comparison"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clara.llm import LLMConfig, MedicalLLM

# Test query
QUERY = "45yo female: fatigue 3mo, weight gain 15lbs, cold intolerance, constipation, dry skin. Diagnosis and tests?"

print("🏥 CLARA - Full LLM Response Comparison")
print("=" * 80)
print(f"\n📝 Query: {QUERY}\n")
print("=" * 80)

# Test both models
models = [
    ("Groq (Llama 3.3 70B)", "groq"),
    ("HuggingFace (Medical-8B)", "huggingface")
]

responses = {}

for name, provider in models:
    print(f"\n🔄 Generating response from {name}...")
    try:
        llm = MedicalLLM(LLMConfig(provider=provider))
        if llm.is_available():
            response = llm.medical_query(QUERY)
            responses[name] = response
            print(f"✅ Response received ({len(response)} characters)")
        else:
            print(f"❌ {name} not available")
    except Exception as e:
        print(f"❌ Error: {e}")

# Display full responses
print("\n" + "=" * 80)
print("📄 FULL RESPONSES")
print("=" * 80)

for name, response in responses.items():
    print(f"\n{'─' * 80}")
    print(f"🤖 {name}")
    print('─' * 80)
    print(response)
    print()

# Summary
print("=" * 80)
print("📊 COMPARISON SUMMARY")
print("=" * 80)
for name, response in responses.items():
    word_count = len(response.split())
    lines = response.count('\n') + 1
    print(f"\n{name}:")
    print(f"  - Length: {len(response)} characters")
    print(f"  - Words: {word_count}")
    print(f"  - Lines: {lines}")
