---
inclusion: auto
---

# CLARA Paper — "Clinical Education Must Evolve"

Submitted to NeurIPS 2025. This steering file summarizes the paper so any work in this workspace stays aligned with what the paper describes.

## Paper Thesis

Clinical education faces a crisis: shortage of supervisors, inconsistent feedback, limited clinical placements (58% of coordinators report difficulty), and a persistent theory-practice gap. Supervised medical professionals show 24% reduced mortality risk vs unsupervised — so supervision quality directly impacts patient outcomes. CLARA is proposed as an AI system that augments (not replaces) human mentorship by providing real-time, standardized, scalable feedback on clinical reasoning.

## What CLARA Is (Per the Paper)

A multimodal AI education system that integrates three data types through a mixture-of-experts architecture:
- **Voice input** — trainee-patient conversations
- **Electronic Health Records (EHR)** — structured patient data (demographics, labs, meds, diagnoses, notes)
- **Medical imaging** — CT, MRI, X-ray, 3D volumetric data

It processes these through specialized modules, compares trainee performance against expert patterns, and delivers real-time feedback.

## Architecture (6 Layers)

### Layer 1: Data Inputs
| Source | What It Provides |
|--------|-----------------|
| Bridge2AI Voice | Clinical speech recordings with speaker role, speech acts, acoustic metadata |
| MIMIC-IV | 70K+ ICU admissions — demographics, labs, meds, diagnoses, clinical notes |
| MedAlign | Curated mapping between medical dialogues and structured EHR fields/codes |

### Layer 2: Data Ingestion
- **ASR Models** — Domain-adapted Automatic Speech Recognition fine-tuned on clinical corpora. Handles medical vocabulary, overlapping speech, ambient noise. Transcribes speech to text.
- **Acoustic extraction** — wav2vec / HuBERT extract speech rate, pitch variation, pauses, tone. Detects stress, confusion, urgency.
- **EHR Integration** — Temporal encoding of structured data using TCN / Transformer encoders. Produces time-aware patient state vectors.
- **Image Processing** — 3D rendering, multi-plane navigation for volumetric imaging data.

### Layer 3: Interpretation Engine
- **Medical NLP** — BioBERT / ClinicalBERT for context-aware embeddings. Intent recognition, named entity extraction (symptoms, medications). Trained on MedAlign to map speech → EHR concepts (e.g., "patient is on metformin" → Medication: Metformin).
- **Medical Concept Extraction** — Identifies clinical concepts from conversation.
- **Image Analysis** — Computer vision for pathology detection in medical images.

### Layer 4: Reasoning Core
- **Semantic Alignment Engine** — Mixture-of-Experts (MoE) framework. Domain-specific expert modules for text, audio, structured data. Cross-modal attention for text-image alignment. Knowledge graphs + medical ontologies encode clinical reasoning principles (hypothesis refinement, differential diagnosis).
- **Clinical Knowledge Database** — Adaptive, self-updating (not static). Links pharmacology, physiology, basic sciences, communication. Evidence-mapped knowledge graphs improve diagnostic accuracy by 27% over single-domain systems. Includes bias mitigation and confidence scoring.

### Layer 5: AI-Generated Feedback
- **Communication Feedback** — Questioning technique (open vs closed), empathy detection, history-taking structure. Uses NLP + acoustic features.
- **Diagnostic Feedback** — Reasoning completeness, missed diagnoses, longitudinal error detection, rare disease awareness. Explainable decision models.
- **Treatment Plan Feedback** — Evidence-based recommendations via LLMs. Ethical and personalized. Clinical safety standards maintained.

### Layer 6: System Response
- **Real-Time Feedback** — During encounter, microsecond resolution. Identifies errors, suggests improvements, reinforces correct decisions. Adapted to user role and experience level.
- **Performance Analytics** — Diagnostic accuracy, treatment appropriateness, decision-making quality. Visual breakdowns. Peer benchmarking. Aligns with Quadruple Aim.
- **Adaptive Learning Insights** — Personalized recommendations from interaction history. Predictive difficulty adjustment. Distinguishes stable knowledge gaps from temporary lapses.

## Three Scenarios

### Scenario 1: Initial History Taking and Diagnostic Reasoning
- Student practices chest pain history-taking
- Voice → ASR → NLP extracts concepts ("chest pain", "radiation to left arm") → Semantic alignment against KB → Identifies omissions (e.g., missed family history) → Communication + Diagnostic feedback → Adaptive insights update learning profile
- **This is what our prototype implements**

### Scenario 2: Advanced 3D Medical Imaging Analysis
- Student interprets CT pulmonary angiogram for suspected PE
- EHR provides clinical context (D-dimer, travel history) → 3D image rendering → Student verbally interprets → NLP extracts radiological concepts ("filling defect", "pulmonary artery branches") → Spatial reasoning assessment → 3D heat map of reviewed vs missed regions
- **Not yet implemented**

### Scenario 3: Longitudinal Patient Management
- Type 2 diabetes patient across 6 months of encounters
- Each visit: new voice input + updated EHR + evolving imaging (diabetic retinopathy) → Tracks trends over time → Evaluates medication adjustments, preventive care, referrals → Communication feedback on patient education and shared decision-making
- **Not yet implemented**

## Key Claims and Numbers from the Paper
- Supervised practice reduces mortality by 24%
- 58% of clinical coordinators report difficulty securing placements
- Knowledge graphs improve diagnostic accuracy by 27% over single-domain systems
- Bias mitigation frameworks show up to 42% reduction in diagnostic disparities
- System processes at microsecond resolution for real-time integration

## Challenges the Paper Acknowledges
1. **Authentic simulation** — Clinical reasoning is contextual, involves subtle cues, non-verbal communication, cultural context. AI struggles with ambiguity and "gray areas" where multiple interpretations are valid. Risk of presenting reasoning as more linear than it truly is. Risk of codifying existing biases.
2. **Ecosystem integration** — Faculty may see AI as threat. Students may "game" the system. Technical integration across diverse LMS and EHR platforms. Undefined sustainable cost models.

## Success Criteria (from Project Proposal)
- Clinical reasoning extraction: >85% F1-score against expert annotations
- Real-time latency: <3 seconds for transcription + feedback
- User satisfaction: >75% positive feedback from trainees and instructors
- Demonstrated improvement in diagnostic performance in simulated encounters
- HIPAA and IRB compliance
- Functional UI deployment in pilot clinical education settings

## Infrastructure (from Project Proposal)
- AWS Transcribe Medical ($7K) — real-time ASR
- Amazon SageMaker ($15K) — model training and hosting
- Bedrock/JumpStart ($10K) — foundation models for reasoning extraction
- EC2 ($13K) — real-time analytics and feedback
- S3 + Monitoring ($5K) — secure storage

## How This Maps to Our Code

| Paper Component | Our Implementation | Status |
|----------------|-------------------|--------|
| ASR Models (wav2vec, HuBERT) | Whisper (VietMed) + Mock/VOSK (simulation) | Partial |
| Acoustic features (pitch, pauses, tone) | Librosa in VietMed.ipynb, mock in asr.py | Partial |
| Medical NLP (BioBERT, ClinicalBERT) | bert-clinical-ner + spaCy + regex | Partial |
| MedAlign dialogue-EHR mapping | Not implemented | ❌ |
| EHR temporal encoding (TCN/Transformer) | Not implemented | ❌ |
| MoE reasoning engine | Not implemented (using if/else + LLM calls) | ❌ |
| Semantic Alignment Engine | semantic.py (concept matching against flat JSON KB) | Basic |
| Clinical Knowledge Database | clinical_knowledge.json with expert patterns | Basic |
| Knowledge graphs / ontologies | Not implemented | ❌ |
| Communication Feedback | feedback.py (open/closed ratio, empathy detection) | ✅ |
| Diagnostic Feedback | feedback.py (coverage, missing concepts, critical gaps) | ✅ |
| Treatment Plan Feedback | Not implemented | ❌ |
| Real-Time Feedback | run_session.py / interactive_session.py | ✅ |
| Performance Analytics | analytics.py (scoring, expert comparison) | ✅ |
| Adaptive Learning | adaptive.py (personalized feedback, learning pathways) | ✅ |
| Medical Imaging (Scenario 2) | Not implemented | ❌ |
| Longitudinal Management (Scenario 3) | Not implemented | ❌ |
| AWS deployment | Not implemented (runs locally) | ❌ |
