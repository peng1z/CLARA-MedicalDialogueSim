---
inclusion: auto
---

# CLARA Project — Steering Document

## What Is This Project

CLARA (Clinical Listening and Reasoning Assistant) is an AI-powered system for real-time feedback on clinical reasoning during medical trainee-patient interactions. It's a research project led by Dr. Divya Chaudhary at Northeastern University (Khoury College), submitted to NeurIPS 2025, with $70K cash + $50K AWS credits funding.

The core idea: listen to a trainee's clinical conversation, transcribe it, extract what clinical concepts they asked about, compare that against what an expert would ask, and give real-time feedback on what they missed, what they did well, and how to improve.

## Workspace Structure

This workspace has two main codebases plus supporting files:

### VietMed.ipynb (Data Analysis Layer)
- Colab notebook that processes real Vietnamese medical audio (VietMed dataset) and English clinical notes (MIMIC-IV)
- Pipeline: Whisper ASR → Google Translate (Vi→En) → Medical NER → Expert vs Trainee comparison
- Outputs: `mimic_ner_results.csv`, `vietmed_ner_results.csv`, `vietmed_features.csv`
- Key finding: experts generate 2.8x more reasoning elements than trainees

### CLARA-MedicalDialogueSim/ (Interactive Simulation)
- Python package implementing Scenario 1: chest pain history-taking
- Pipeline: ASR → NLP concept extraction → Semantic alignment against KB → Feedback → Analytics → Adaptive learning
- Two modes: rule-based (no dependencies) and LLM-enhanced (OpenRouter with Deepseek R1 + Qwen3)
- Entry points: `scripts/run_session.py` (batch), `scripts/interactive_session.py` (interactive)

### Reference Documents
- `642_Clinical_Education_Must_Ev-2.pdf` — NeurIPS 2025 paper describing full CLARA architecture
- `CLARA project proposal.pdf` — Funding proposal with objectives, methods, timeline, budget
- `CLARA_Block_Diagrams.html` — Visual diagrams of paper architecture, both codebases, and integration plan

## What Has Been Done

### Week 1 (VietMed.ipynb)
- Processed 85 Vietnamese medical audio clips through Whisper ASR
- Extracted acoustic features (pitch, tempo, jitter, pauses) with librosa
- Built Vietnamese→English translation pipeline (Google Translate with retry/caching)
- Ran medical NER on 500 MIMIC-IV clinical notes (expert baseline) and 82 VietMed transcripts (trainee data)
- NER model: `samrawal/bert-base-uncased_clinical-ner` via HuggingFace
- Built enhanced reasoning pattern extraction (bilingual regex for questions + hypotheses)
- Created 0-100 scoring framework with A-F grades
- Created personalized feedback generation system

### CLARA-MedicalDialogueSim Prototype
- Built complete Scenario 1 pipeline (chest pain history-taking)
- Modules: `asr.py`, `nlp.py`, `semantic.py`, `feedback.py`, `analytics.py`, `adaptive.py`
- Clinical knowledge base with 12 required concepts for chest pain
- LLM integration with multiple providers (Groq, HuggingFace, OpenAI, Together, Ollama)
- Rule-based fallback for all LLM-dependent features

### Tier 1 Integration (PR #2)
Branch: `feat/tier1-vietmed-integration`

1. **NER model in nlp.py** — Added HuggingFace clinical-ner model with 3-tier extraction (Transformer → spaCy → regex fallback). Added NER-to-KB concept mapping. Added reasoning pattern extraction.
2. **Scoring in analytics.py** — Added `compute_reasoning_quality_score()` (0-100, A-F grades) and `compare_to_expert_baseline()` with MIMIC-IV constants.
3. **Expert baselines in clinical_knowledge.json** — Added 33 expert questions + 30 expert hypotheses from MIMIC-IV analysis, plus expert metrics.
4. **Personalized feedback in adaptive.py** — Added `generate_personalized_feedback()` with strengths, improvement areas, expert examples, and 3-phase learning pathways.

## How to Test

### Integration test (validates all Tier 1 tasks):
```bash
cd CLARA-MedicalDialogueSim
pip install transformers torch  # optional, for NER model
python scripts/test_integration.py
```
The test runs without transformers too — NER tests are skipped gracefully, and rule-based extraction still works.

### Run the simulation:
```bash
cd CLARA-MedicalDialogueSim
python scripts/run_session.py           # batch mode with sample transcript
python scripts/interactive_session.py   # interactive mode
```

### Run with LLM mode:
```bash
# Create .env with OPENROUTER_API_KEY=your_key
cd CLARA-MedicalDialogueSim
python scripts/run_session.py   # select option 2 for LLM-enhanced
```

## What's Next

### Tier 2 — Medium Effort (3-6 weeks)
- [ ] **Acoustic features in ASR**: Replace mock paralinguistics in `asr.py` with real librosa extraction (pitch, jitter, pauses) from VietMed notebook
- [ ] **Better language model**: Replace Google Translate with ClinicalBERT embeddings for medical-aware NLP
- [ ] **Real-time expert comparison**: Load MIMIC baselines at startup, compare during session (not just post-hoc)
- [ ] **Unify codebases**: Single package with shared data layer, consistent imports, one entry point

### Tier 3 — Heavy Lifts (research-level, 2-6 months each)
- [ ] Mixture-of-Experts (MoE) reasoning engine
- [ ] MedAlign dataset integration (dialogue ↔ EHR mapping)
- [ ] Medical imaging pipeline (Scenario 2: 3D CT/MRI)
- [ ] Longitudinal patient management (Scenario 3: multi-encounter)
- [ ] Knowledge graphs with SNOMED-CT/UMLS
- [ ] Cross-modal attention fusion (text + audio + structured data)
- [ ] AWS deployment (Transcribe Medical, SageMaker, Bedrock, EC2, S3)

### Paper Scenarios Status
| Scenario | Status |
|----------|--------|
| 1. History Taking & Diagnostic Reasoning (chest pain) | ✅ Prototype complete, Tier 1 integrated |
| 2. Advanced 3D Medical Imaging Analysis | ❌ Not started |
| 3. Longitudinal Patient Management | ❌ Not started |

## Key Technical Decisions

- **NER model**: `samrawal/bert-base-uncased_clinical-ner` — chosen because VietMed.ipynb validated it at 91.5% confidence on MIMIC-IV data
- **Concept extraction strategy**: 3-tier (Transformer → spaCy → regex) so the system works even without GPU/transformers installed
- **Scoring**: 4-component system (Volume/Diversity/Confidence/Density) rather than simple weighted average, because VietMed analysis showed experts differ from trainees across multiple dimensions
- **Expert baselines**: Hardcoded from MIMIC-IV analysis (33 questions, 85 hypotheses, 0.9151 confidence) rather than computed at runtime, for consistency and speed
- **LLM providers**: Multiple supported (Groq, HuggingFace, OpenAI, Together, Ollama) with rule-based fallback for all features

## Important Files

| File | Purpose |
|------|---------|
| `CLARA-MedicalDialogueSim/src/clara/nlp.py` | NLP + NER + concept extraction |
| `CLARA-MedicalDialogueSim/src/clara/semantic.py` | KB alignment + expert patterns |
| `CLARA-MedicalDialogueSim/src/clara/feedback.py` | Communication + diagnostic feedback |
| `CLARA-MedicalDialogueSim/src/clara/analytics.py` | Scoring + expert comparison |
| `CLARA-MedicalDialogueSim/src/clara/adaptive.py` | Personalized feedback + learning pathways |
| `CLARA-MedicalDialogueSim/src/clara/asr.py` | ASR (mock + VOSK) |
| `CLARA-MedicalDialogueSim/src/clara/llm.py` | Multi-provider LLM client |
| `CLARA-MedicalDialogueSim/data/clinical_knowledge.json` | Knowledge base + expert baselines |
| `CLARA-MedicalDialogueSim/scripts/test_integration.py` | Integration test suite |
| `VietMed.ipynb` | Data analysis notebook |
| `progress.md` | Week-by-week progress report |
| `CLARA_Block_Diagrams.html` | Architecture diagrams |
