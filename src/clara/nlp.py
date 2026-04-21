"""Medical NLP and concept extraction.

This module provides three extraction strategies (tried in order):
1. **Transformer NER** — HuggingFace clinical-ner model (from VietMed integration)
2. **spaCy PhraseMatcher** — keyword-based matching with spaCy
3. **Rule-based fallback** — simple string matching

Also exposes:
- classify_question (open vs closed)
- detect_empathy
- extract_medical_entities (raw NER entities with confidence scores)
- reasoning pattern extraction (questions + hypotheses from text)

Optional LLM integration is available via analyze_utterance_with_llm().
"""
from typing import Dict, List, Tuple, Optional, Any
import re
from pathlib import Path

# --- Transformer NER (from VietMed.ipynb integration) ---
try:
    from transformers import pipeline as hf_pipeline
    _HF_NER_AVAILABLE = True
except ImportError:
    _HF_NER_AVAILABLE = False

_ner_pipeline = None  # lazy-loaded singleton


def _get_ner_pipeline():
    """Lazy-load the HuggingFace clinical NER model (same as VietMed.ipynb)."""
    global _ner_pipeline
    if _ner_pipeline is None and _HF_NER_AVAILABLE:
        try:
            _ner_pipeline = hf_pipeline(
                'token-classification',
                model='samrawal/bert-base-uncased_clinical-ner',
                aggregation_strategy='simple'
            )
        except Exception:
            pass
    return _ner_pipeline


# Mapping from NER entity text → KB concept names
_NER_TO_CONCEPT = {
    # problem entities
    "chest pain": "chest_pain", "angina": "chest_pain", "chest discomfort": "chest_pain",
    "pain": "chest_pain",
    "shortness of breath": "shortness_of_breath", "dyspnea": "shortness_of_breath",
    "breathless": "shortness_of_breath",
    "nausea": "nausea_vomiting", "vomiting": "nausea_vomiting", "emesis": "nausea_vomiting",
    "diaphoresis": "diaphoresis", "sweating": "diaphoresis",
    "hypertension": "past_medical_history", "diabetes": "past_medical_history",
    "htn": "past_medical_history", "dm": "past_medical_history",
    "obesity": "past_medical_history", "copd": "past_medical_history",
    # treatment entities
    "aspirin": "medications", "metformin": "medications", "lisinopril": "medications",
    "beta blocker": "medications", "statin": "medications", "nitroglycerin": "medications",
    # test entities
    "ecg": "past_medical_history", "ekg": "past_medical_history",
    "troponin": "past_medical_history", "cardiac catheterization": "past_medical_history",
}


def extract_medical_entities(text: str, max_length: int = 512) -> List[Dict]:
    """
    Extract raw medical entities using the clinical NER model (from VietMed.ipynb).

    Returns list of dicts with keys: word, entity_group, score.
    Falls back to empty list if model unavailable.
    """
    ner = _get_ner_pipeline()
    if ner is None or not text or len(text.strip()) < 5:
        return []
    try:
        truncated = text[:max_length]
        entities = ner(truncated)
        return [
            {
                "word": e.get("word", ""),
                "entity_group": e.get("entity_group", ""),
                "score": round(e.get("score", 0.0), 4),
            }
            for e in entities
        ]
    except Exception:
        return []


def _ner_entities_to_concepts(entities: List[Dict]) -> List[str]:
    """Map NER entity text to KB concept names."""
    found = set()
    for ent in entities:
        word = ent.get("word", "").lower().strip().strip("#")
        # Direct lookup
        if word in _NER_TO_CONCEPT:
            found.add(_NER_TO_CONCEPT[word])
        else:
            # Fuzzy: check if any mapping key is contained in the entity word
            for key, concept in _NER_TO_CONCEPT.items():
                if key in word or word in key:
                    found.add(concept)
                    break
    return sorted(found)


# --- Reasoning pattern extraction (from VietMed.ipynb enhanced patterns) ---

_QUESTION_PATTERNS = [
    re.compile(r'(?:did|have|any|do you|have you|are you|is there).*?\?', re.I),
    re.compile(r'(?:assess|evaluate|check|examine).*?(?:\?|for)', re.I),
    re.compile(r'(?:what|how|when|where|why|which).*?\?', re.I),
    re.compile(r'(?:can|could|should|would|will).*?\?', re.I),
    re.compile(r'(?:so|then|now|well).*?\?', re.I),
    re.compile(r'(?:symptoms|condition|treatment|medicine).*?\?', re.I),
]

_HYPOTHESIS_PATTERNS = [
    re.compile(r'(?:likely|probably|suspect|think|consider|rule out).*?(?:\.|,)', re.I),
    re.compile(r'differential.*?(?:includes|diagnosis)', re.I),
    re.compile(r'(?:diagnosis|condition|disease).*?(?:is|was|could be).*?(?:\.|,)', re.I),
    re.compile(r'(?:maybe|perhaps|possibly|might be|could be).*?(?:\.|,)', re.I),
    re.compile(r'(?:I think|we think|it seems|appears to be).*?(?:\.|,)', re.I),
    re.compile(r'(?:should|need to|have to|must).*?(?:treat|check|test).*?(?:\.|,)', re.I),
    re.compile(r'(?:because|since|due to|caused by).*?(?:\.|,)', re.I),
]


def extract_reasoning_patterns(text: str) -> Dict[str, List[str]]:
    """
    Extract clinical reasoning patterns (questions and hypotheses) from text.
    Uses enhanced patterns from VietMed.ipynb integration.
    """
    questions = []
    for pat in _QUESTION_PATTERNS:
        questions.extend(pat.findall(str(text)))

    hypotheses = []
    for pat in _HYPOTHESIS_PATTERNS:
        hypotheses.extend(pat.findall(str(text)))

    return {"questions": questions, "hypotheses": hypotheses}


# --- Original imports ---

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

try:
    from .llm import LLMClient
    _LLM_AVAILABLE = True
except (ImportError, Exception):
    _LLM_AVAILABLE = False
    LLMClient = Any  # type: ignore

# Small keyword-to-concept mapping for chest pain scenario
_CONCEPT_KEYWORDS = {
    "chest_pain": ["chest pain", "pain in the chest", "central chest pain"],
    "radiation": ["left arm", "jaw", "back", "radiat"],
    "shortness_of_breath": ["shortness of breath", "short of breath", "sob", "breathless", "dyspn"],
    "diaphoresis": ["sweat", "diaphoresis", "clammy"],
    "nausea_vomiting": ["nausea", "vomit"],
    "onset_timing": ["when did", "onset", "how long"],
    "quality": ["sharp", "dull", "pressure", "tightness", "crushing"],
    "severity": ["severe", "mild", "scale of 1 to 10", "how bad"],
    "past_medical_history": ["history of", "past medical", "pmh", "hypertension", "diabetes", "high blood"],
    "family_history": ["family history", "father had", "mother had", "siblings"],
    "smoking": ["smoke", "smoking", "tobacco"],
    "medications": ["medication", "taking", "aspirin", "beta blocker"],
}

_OPEN_Q_PATTERNS = re.compile(r"^(what|how|tell|describe|could you tell|please describe|can you tell)", re.I)
_CLOSED_Q_PATTERNS = re.compile(r"^(do|did|have|has|is|are|was|were|can|could|would|should)\\b", re.I)


def _build_spacy_matcher(nlp):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns_by_concept = {}
    for concept, kws in _CONCEPT_KEYWORDS.items():
        patterns = [nlp.make_doc(kw) for kw in kws]
        matcher.add(concept, patterns)
        patterns_by_concept[concept] = patterns
    return matcher


def classify_question(text: str) -> str:
    t = text.strip()
    if not t:
        return "unknown"
    if _OPEN_Q_PATTERNS.search(t):
        return "open"
    if _CLOSED_Q_PATTERNS.search(t):
        return "closed"
    # fallback: treat WH- words as open
    if any(t.lower().startswith(w) for w in ["who", "what", "when", "where", "why", "how"]):
        return "open"
    return "closed"


def extract_concepts(text: str) -> List[str]:
    """
    Extract clinical concepts from text. Tries strategies in order:
    1. Transformer NER model (clinical-ner from VietMed integration)
    2. spaCy PhraseMatcher
    3. Rule-based keyword matching
    """
    text_low = text.lower()
    found = set()

    # Strategy 1: Transformer NER model (from VietMed.ipynb)
    if _HF_NER_AVAILABLE:
        try:
            entities = extract_medical_entities(text)
            ner_concepts = _ner_entities_to_concepts(entities)
            if ner_concepts:
                found.update(ner_concepts)
        except Exception:
            pass

    # Strategy 2: spaCy PhraseMatcher
    if _SPACY_AVAILABLE:
        try:
            nlp = getattr(extract_concepts, "_nlp", None)
            matcher = getattr(extract_concepts, "_matcher", None)
            if nlp is None:
                nlp = spacy.load("en_core_web_sm")
                matcher = _build_spacy_matcher(nlp)
                extract_concepts._nlp = nlp
                extract_concepts._matcher = matcher

            doc = nlp(text)
            matches = matcher(doc)
            for match_id, start, end in matches:
                concept = nlp.vocab.strings[match_id]
                found.add(concept)
        except Exception:
            pass

    # Strategy 3: Rule-based fallback (always runs to catch what NER misses)
    for concept, keywords in _CONCEPT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_low:
                found.add(concept)
                break

    return sorted(found)


def detect_empathy(text: str) -> bool:
    # Very simple heuristic: presence of phrases expressing concern
    empathy_phrases = ["how are you feeling", "i'm sorry", "that must be", "that sounds", "i understand"]
    t = text.lower()
    return any(p in t for p in empathy_phrases)


def analyze_utterance(text: str) -> Dict:
    qtype = classify_question(text)
    concepts = extract_concepts(text)
    empathy = detect_empathy(text)
    entities = extract_medical_entities(text)
    reasoning = extract_reasoning_patterns(text)
    return {
        "text": text,
        "type": qtype,
        "concepts": concepts,
        "empathy": empathy,
        "medical_entities": entities,
        "reasoning_patterns": reasoning,
    }


def analyze_utterance_with_llm(
    text: str,
    client: Optional[LLMClient] = None,
    use_llm_fallback: bool = False
) -> Dict:
    """
    Analyze a medical utterance using LLM (with rule-based fallback).
    
    Args:
        text: The utterance to analyze
        client: LLMClient instance. If None, creates one using OPENROUTER_API_KEY env var.
        use_llm_fallback: If True, falls back to rule-based analysis if LLM fails.
                         If False, raises exception on LLM failure.
    
    Returns:
        Dict with analysis including text, type, concepts, empathy, and llm_analysis
    """
    if not _LLM_AVAILABLE:
        if use_llm_fallback:
            return analyze_utterance(text)
        raise ImportError("LLM integration not available. Install python-dotenv and requests packages or set use_llm_fallback=True.")
    
    if client is None:
        client = LLMClient()
    
    try:
        # Get LLM analysis
        llm_result = client.analyze_medical_utterance(text)
        
        # Get rule-based analysis for comparison
        rule_analysis = analyze_utterance(text)
        
        # Merge results: use LLM concepts if available, fall back to rule-based
        concepts = llm_result.get("concepts", [])
        if isinstance(concepts, str):
            concepts = [c.strip() for c in concepts.split(",")]
        if not concepts:
            concepts = rule_analysis.get("concepts", [])
        
        qtype = llm_result.get("question_type", rule_analysis.get("type", "unknown"))
        if qtype not in ["open", "closed", "statement", "unclear"]:
            qtype = rule_analysis.get("type", qtype)
        
        return {
            "text": text,
            "type": qtype,
            "concepts": concepts,
            "empathy": rule_analysis.get("empathy", False),
            "llm_analysis": llm_result,
            "clinical_relevance": llm_result.get("clinical_relevance", "medium"),
        }
    except Exception as e:
        if use_llm_fallback:
            # Fall back to rule-based analysis
            return analyze_utterance(text)
        raise
