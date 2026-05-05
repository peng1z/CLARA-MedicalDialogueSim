"""Interactive session CLI for CLARA prototype.

Supports two modes:
- text: type utterances (blank line to finish)
- audio: provide a WAV file path to transcribe with local VOSK model (optional)

The script processes utterances through the pipeline and prints real-time and post-encounter feedback.
"""
import sys
from pathlib import Path

# ensure src on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from clara.asr import transcribe, transcribe_audio
from clara.nlp import analyze_utterance, analyze_utterance_with_llm
from clara.semantic import align_concepts_to_kb
from clara.feedback import communication_feedback, diagnostic_feedback, enhanced_feedback_with_llm
from clara.analytics import compute_score
from clara.adaptive import generate_adaptive_insights, generate_adaptive_insights_with_llm
from clara.llm import LLMClient


def interactive_text_mode(student_id: str = "student_001", use_llm: bool = False):
    print("Interactive text mode. Type each student utterance and press Enter. Blank line to finish.")
    analyses = []
    asked_concepts_sequence = []
    
    # Initialize LLM client if needed
    llm_client = None
    if use_llm:
        try:
            llm_client = LLMClient()
            print(f"[LLM Mode] Using OpenRouter models for analysis\n")
        except ValueError as e:
            print(f"[Warning] LLM not available: {e}")
            print("Falling back to rule-based analysis\n")
            use_llm = False
    
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        asr_out = transcribe(line)
        
        if use_llm and llm_client:
            analysis = analyze_utterance_with_llm(asr_out["text"], client=llm_client, use_llm_fallback=True)
        else:
            analysis = analyze_utterance(asr_out["text"])
        analyses.append(analysis)
        for c in analysis.get("concepts", []):
            if c not in asked_concepts_sequence:
                asked_concepts_sequence.append(c)

    # post processing
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


def audio_file_mode(student_id: str = "student_001", use_llm: bool = False):
    wav = input("Enter path to WAV file (PCM 16k mono): ").strip()
    if not wav:
        print("No file provided. Exiting.")
        return
    
    # Initialize LLM client if needed
    llm_client = None
    if use_llm:
        try:
            llm_client = LLMClient()
        except ValueError as e:
            print(f"[Warning] LLM not available: {e}")
            use_llm = False
    
    try:
        asr_out = transcribe_audio(wav)
    except Exception as e:
        print("ASR error:", e)
        return

    print("Transcribed text:", asr_out.get("text"))
    
    if use_llm and llm_client:
        analysis = analyze_utterance_with_llm(asr_out.get("text", ""), client=llm_client, use_llm_fallback=True)
    else:
        analysis = analyze_utterance(asr_out.get("text", ""))
    
    semantic_result = align_concepts_to_kb(analysis.get("concepts", []))
    
    if use_llm and llm_client:
        feedback = enhanced_feedback_with_llm([analysis], semantic_result, client=llm_client, use_llm_fallback=True)
        diag_fb = feedback.get("diagnostic_feedback", {})
        comm_fb = feedback.get("communication_feedback", {})
    else:
        comm_fb = communication_feedback([analysis])
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

    print('\n=== Post-Encounter Report ===')
    print('Asked concepts sequence:', analysis.get('concepts', []))
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


def main():
    print("=" * 70)
    print("CLARA Interactive Session")
    print("=" * 70)
    print("\nSelect analysis mode:")
    print("1. Rule-based (fast, deterministic)")
    print("2. LLM-enhanced (uses OpenRouter models for better insights)")
    
    mode_choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    use_llm = mode_choice == "2"
    
    if use_llm:
        print("\n[Mode: LLM-Enhanced]")
    else:
        print("\n[Mode: Rule-Based]")
    
    print("\nSelect input mode:")
    print("1. Text (type utterances)")
    print("2. Audio (WAV file)")
    
    input_mode = input("\nEnter choice (1 or 2, default=1): ").strip().lower()
    if input_mode == "2":
        audio_file_mode(use_llm=use_llm)
    else:
        interactive_text_mode(use_llm=use_llm)


if __name__ == '__main__':
    main()
