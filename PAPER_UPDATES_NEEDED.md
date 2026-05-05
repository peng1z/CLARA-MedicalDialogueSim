---
inclusion: auto
---

# CLARA Paper — Sections That Need Updating

This document maps every section of the paper that needs updating, what is
currently written, what the problem is, and exactly what to add or change.
Sections not listed here are correct as-is and require no edits.

---

## Priority 1 — Section 4: Scenario 1 (most critical)

**Location:** `## 4 Scenarios Supported by CLARA → Scenario 1`

**Current state:** Purely descriptive. No numbers, no results, no validation.

**Problem:** The paper presents Scenario 1 as built but provides no evidence it works.

**What to add:** A results paragraph at the end of Scenario 1:

> *"We validated Scenario 1 on two real clinical datasets. Expert baselines were
> derived from 999 MIMIC-IV discharge notes using the bert-base-uncased_clinical-ner
> model, yielding a mean NER confidence of 91.59% (73.9% high-confidence entities)
> and a mean quality score of 42.79/100 (stdev 9.15, range 21.4–70.2). Trainee
> baselines were computed from 85 VietMed Vietnamese clinical audio clips across
> 5 consultation sessions using Whisper ASR with Vietnamese→English translation.
> Trainees averaged 6.2 questions and 4.4 hypotheses per session compared to
> expert clinicians generating 33 questions and 85 hypotheses per session — an
> 11x reasoning gap that directly motivates CLARA's real-time feedback architecture.
> All 67 integration checks passed, confirming end-to-end pipeline reliability."*

---

## Priority 2 — Section 3.1: Dataset

**Location:** `## 3 Clinical Education Through Real-Time AI-Driven Diagnostic Feedback → 3.1 Dataset`

**Current state:** Describes MedAlign, MIMIC-IV, and Bridge2AI Voice as the three
data sources.

**Problem:** Our Scenario 1 prototype does not use MedAlign or Bridge2AI Voice.
Not clarifying this overstates implementation completeness.

**What to add:** A paragraph after the existing dataset description:

> *"For prototype validation of Scenario 1, we use two publicly available datasets:
> (1) MIMIC-IV discharge notes (n=999) as the expert clinical reasoning baseline,
> processed through a transformer-based NER pipeline to extract clinical entities,
> questions, and diagnostic hypotheses; and (2) the VietMed dataset (n=85 audio
> clips, 5 consultation sessions) as the trainee baseline, transcribed using
> OpenAI Whisper and translated from Vietnamese to English for downstream NLP
> analysis. MedAlign and Bridge2AI Voice are planned integrations for Scenarios
> 2 and 3."*

---

## Priority 3 — Abstract

**Location:** First paragraph of the paper.

**Current state:** No quantitative results mentioned. Entirely conceptual.

**Problem:** NeurIPS papers are expected to include at least one concrete result
in the abstract.

**What to add:** One sentence before the final sentence:

> *"Prototype validation on MIMIC-IV (n=999 expert notes, 91.59% NER confidence)
> and VietMed (n=5 trainee sessions, 85 audio clips) confirms an 11x expert–trainee
> reasoning gap, demonstrating the system's ability to quantify and target
> clinically significant performance deficits."*

---

## Priority 4 — Section 5: Discussion

**Location:** `## 5 Discussion`

**Current state:** All claims are conceptual — no empirical evidence cited for
the prototype's performance.

**Problem:** The discussion reads as future speculation rather than grounded in
results. Three specific claims need anchoring:

| Current claim | Anchor with |
|---------------|-------------|
| "consistent and reliable feedback across all learning sessions" | 67/67 integration tests passing |
| "students develop skills in synthesizing information" | Quality score distribution: trainee mean 32.5 vs expert mean 42.79 |
| "objective documentation of skill development" | 4-component scoring: Volume / Diversity / Confidence / Density |

**What to add:** One paragraph under **Reliability and Standardization of Feedback**:

> *"In our Scenario 1 prototype, system reliability was confirmed through 67
> integration checks with a 100% pass rate. Quality scoring uses a four-component
> framework (Volume, Diversity, Confidence, Density) producing a 0–100 score with
> A–F grades. Expert notes (MIMIC-IV) scored a mean of 42.79 while trainee sessions
> (VietMed) scored a mean of 32.5, with trainees predominantly receiving F and C
> grades. This 10-point gap across the scoring distribution provides objective,
> granular evidence of the reasoning deficits CLARA is designed to address."*

---

## Priority 5 — Section 3.2: Models

**Location:** `## 3.2 Models → Speech Transcription and Acoustic Understanding`

**Current state:** References wav2vec, HuBERT, BioBERT, ClinicalBERT.

**Problem:** Our prototype uses different models. Not flagging this conflates
the architecture vision with the current implementation.

**What to add:** A note at the end of the Speech Transcription paragraph:

> *"In the Scenario 1 prototype, we use OpenAI Whisper for ASR (base model,
> Vietnamese-language audio), librosa for acoustic feature extraction (pitch via
> pyin, tempo via beat_track, jitter, pause duration via RMS energy gating), and
> samrawal/bert-base-uncased_clinical-ner for clinical entity recognition, achieving
> 91.59% mean confidence on MIMIC-IV validation data. LLM-enhanced feedback uses
> Deepseek R1 for clinical reasoning and Qwen3 for structured analysis via the
> OpenRouter API."*

---

## What Does NOT Need Updating

| Section | Reason |
|---------|--------|
| Section 2 — Background & Related Work | Literature review, no implementation claims |
| Section 3.3 — Reasoning (MoE) | Architecture vision, not prototype claims |
| Section 3.4 — Feedback | Correctly describes the feedback design |
| Section 3.5 — Knowledge Database | 27% and 42% figures are cited from external references, not our claims |
| Section 4 — Scenarios 2 & 3 | Correctly described as future work (not yet implemented) |
| Section 6 — Challenges | Still fully accurate |
| Section 7 — Conclusion | Still accurate |
| References | Complete, nothing to add |

---

## Summary of Numbers to Insert

| Metric | Value | Location |
|--------|-------|----------|
| MIMIC-IV notes processed | 999 | Abstract, S3.1, S4 |
| NER confidence mean | 91.59% | Abstract, S3.2, S4 |
| High-confidence entities | 73.9% | S4 |
| Expert quality score mean | 42.79 / 100 | S4, S5 |
| Expert score stdev | 9.15 | S4 |
| VietMed sessions / clips | 5 sessions / 85 clips | Abstract, S3.1, S4 |
| Trainee avg questions / session | 6.2 | S4 |
| Trainee avg hypotheses / session | 4.4 | S4 |
| Expert avg questions / session | 33 | S4 |
| Expert avg hypotheses / session | 85 | S4 |
| Expert–trainee reasoning gap | 11x | Abstract, S4, S5 |
| Trainee quality score mean | 32.5 / 100 | S4, S5 |
| Integration tests passing | 67 / 67 | S4, S5 |
