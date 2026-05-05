"""Demo runner for Scenario 1 - Initial History Taking and Diagnostic Reasoning

This script simulates a student asking questions (from data/sample_transcript.txt),
processes them through the mock ASR -> NLP -> Semantic -> Feedback pipeline,
and prints real-time feedback & a post-encounter report.
"""
import sys
from pathlib import Path

# ensure src is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from clara.asr import transcribe
from clara.nlp import analyze_utterance, analyze_utterance_with_llm
from clara.semantic import align_concepts_to_kb
from clara.feedback import communication_feedback, diagnostic_feedback, enhanced_feedback_with_llm
from clara.analytics import compute_score
from clara.adaptive import generate_adaptive_insights, generate_adaptive_insights_with_llm
from clara.llm import LLMClient


def load_transcript(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def run_simulation(student_id: str = "student_001", use_llm: bool = False):
    transcript_path = REPO_ROOT / "data" / "sample_transcript.txt"
    utterances = load_transcript(transcript_path)

    analyses = []
    asked_concepts_sequence = []
    
    # Initialize LLM client if needed
    llm_client = None
    if use_llm:
        try:
            llm_client = LLMClient()
            print(f"[LLM Mode] Using {llm_client.model} for reasoning and {llm_client.STRUCTURED_MODEL} for analysis\n")
        except ValueError as e:
            print(f"[Warning] LLM not available: {e}")
            print("Falling back to rule-based analysis\n")
            use_llm = False

    # Real-time rule: if after 4 utterances critical chest pain details are missing, prompt real-time feedback
    for idx, utt in enumerate(utterances, start=1):
        # ASR (mock)
        asr_out = transcribe(utt)
        text = asr_out["text"]

        # Medical NLP
        if use_llm and llm_client:
            analysis = analyze_utterance_with_llm(text, client=llm_client, use_llm_fallback=True)
        else:
            analysis = analyze_utterance(text)
        analyses.append(analysis)

        # track sequence of concepts as they are asked
        for c in analysis.get("concepts", []):
            if c not in asked_concepts_sequence:
                asked_concepts_sequence.append(c)

        # Real-time check after a few initial questions
        if idx == 4:
            semantic_partial = align_concepts_to_kb(asked_concepts_sequence)
            if semantic_partial.get("critical_missing"):
                print("[REAL-TIME FEEDBACK] Critical red-flag items not yet asked:", ", ".join(semantic_partial.get("critical_missing")))
                print("Consider asking about onset, quality, radiation, and shortness of breath.")

    # Post-encounter processing
    semantic_result = align_concepts_to_kb(asked_concepts_sequence)
    
    if use_llm and llm_client:
        feedback = enhanced_feedback_with_llm(analyses, semantic_result, client=llm_client, use_llm_fallback=True)
        diag_fb = feedback.get("diagnostic_feedback", {})
        comm_fb = feedback.get("communication_feedback", {})
    else:
        comm_fb = communication_feedback(analyses)
        diag_fb = diagnostic_feedback(semantic_result)
    
    score = compute_score(diag_fb, comm_fb)
    
    if use_llm and llm_client:
        adaptive = generate_adaptive_insights_with_llm(
            student_id,
            diag_fb,
            comm_fb,
            client=llm_client,
            use_llm_fallback=True
        )
    else:
        adaptive = generate_adaptive_insights(student_id, diag_fb, comm_fb)

    # Print report
    print('\n=== Post-Encounter Report ===')
    print('Asked concepts sequence:', asked_concepts_sequence)
    print('\n-- Communication Feedback --')
    print(comm_fb.get('notes'))
    print('\n-- Diagnostic Feedback --')
    print('Coverage: {:.1%}'.format(diag_fb.get('coverage', 0)))
    for s in diag_fb.get('suggestions', []):
        print('-', s)
    print('\n-- Score --')
    print('Score:', score.get('score'), '(peer avg: {})'.format(score.get('peer_avg')))
    print('\n-- Adaptive Insights --')
    print('Recommended next case:', adaptive.get('recommended_next_case'))


if __name__ == '__main__':
    # Ask user which mode to use
    print("=" * 70)
    print("CLARA Medical Dialogue Simulation Pipeline")
    print("=" * 70)
    print("\nSelect analysis mode:")
    print("1. Rule-based (fast, deterministic)")
    print("2. LLM-enhanced (uses OpenRouter models for better insights)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    use_llm = choice == "2"
    
    if use_llm:
        print("\n[Mode: LLM-Enhanced]")
    else:
        print("\n[Mode: Rule-Based]")
    
    print()
    run_simulation(use_llm=use_llm)
