"""LLM Provider Module for CLARA Medical Dialogue System."""
import os
import re
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "groq"
    model: str = None
    api_key: str = None
    base_url: str = None
    temperature: float = 0.3
    max_tokens: int = 1024
    
    DEFAULT_MODELS = {
        "groq": "llama-3.3-70b-versatile",
        "huggingface": "Intelligent-Internet/II-Medical-8B",
        "openai": "gpt-4o-mini",
        "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "ollama": "llama3.1:8b",
    }
    
    def __post_init__(self):
        if self.model is None:
            self.model = self.DEFAULT_MODELS.get(self.provider, "llama-3.3-70b-versatile")
        if self.api_key is None:
            env_keys = {"groq": "GROQ_API_KEY", "huggingface": "HF_API_KEY", 
                       "openai": "OPENAI_API_KEY", "together": "TOGETHER_API_KEY"}
            if env_var := env_keys.get(self.provider):
                self.api_key = os.getenv(env_var)


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class GroqProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        try:
            from groq import Groq
            if config.api_key:
                self.client = Groq(api_key=config.api_key)
        except ImportError:
            pass
    
    def is_available(self) -> bool:
        return self.client is not None and self.config.api_key is not None
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Groq not available. Set GROQ_API_KEY.")
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content


class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        try:
            from huggingface_hub import InferenceClient
            if config.api_key:
                self.client = InferenceClient(provider="featherless-ai", api_key=config.api_key)
        except ImportError:
            pass
    
    def is_available(self) -> bool:
        return self.client is not None and self.config.api_key is not None
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("HuggingFace not available. Set HF_API_KEY.")
        
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[-1]["role"] == "user":
            enhanced_messages[-1]["content"] += "\n\nPlease reason step-by-step, and put your final answer within \\boxed{}."
        
        try:
            completion = self.client.chat.completions.create(
                model=kwargs.get("model", self.config.model),
                messages=enhanced_messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            content = completion.choices[0].message.content
            if match := re.search(r'\\boxed\{([^}]+)\}', content):
                content = content.replace(f'\\boxed{{{match.group(1)}}}', f'\n\n**Final Answer:** {match.group(1)}')
            return content
        except Exception as e:
            raise RuntimeError(f"HuggingFace API error: {e}")


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        try:
            from openai import OpenAI
            if config.api_key:
                self.client = OpenAI(api_key=config.api_key)
        except ImportError:
            pass
    
    def is_available(self) -> bool:
        return self.client is not None and self.config.api_key is not None
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI not available. Set OPENAI_API_KEY.")
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content


class TogetherProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        try:
            from openai import OpenAI
            if config.api_key:
                self.client = OpenAI(api_key=config.api_key, base_url="https://api.together.xyz/v1")
        except ImportError:
            pass
    
    def is_available(self) -> bool:
        return self.client is not None and self.config.api_key is not None
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Together not available. Set TOGETHER_API_KEY.")
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content


class OllamaProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
    
    def is_available(self) -> bool:
        try:
            import requests
            return requests.get(f"{self.base_url}/api/tags", timeout=2).status_code == 200
        except:
            return False
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import requests
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": kwargs.get("model", self.config.model), "messages": messages, "stream": False,
                  "options": {"temperature": kwargs.get("temperature", self.config.temperature)}},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class MedicalLLM:
    """High-level interface for medical dialogue LLM interactions."""
    
    SYSTEM_PROMPT = """You are a medical education assistant for CLARA (Clinical Learning and Real-time Assessment).
Your role is to help medical students practice patient interviews and clinical reasoning.
Provide accurate, evidence-based medical information. This is for educational purposes only."""

    PROVIDERS = {
        "groq": GroqProvider, "huggingface": HuggingFaceProvider, "hf": HuggingFaceProvider,
        "openai": OpenAIProvider, "together": TogetherProvider, "ollama": OllamaProvider,
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        provider_class = self.PROVIDERS.get(self.config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.config.provider}")
        self.provider = provider_class(self.config)
        self.conversation_history: List[Dict[str, str]] = []
    
    def is_available(self) -> bool:
        return self.provider.is_available()
    
    def reset_conversation(self):
        self.conversation_history = []
    
    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": system_prompt or self.SYSTEM_PROMPT}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        response = self.provider.generate(messages)
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    def medical_query(self, query: str, context: Optional[str] = None) -> str:
        system = self.SYSTEM_PROMPT + (f"\n\nContext: {context}" if context else "")
        return self.provider.generate([{"role": "system", "content": system}, {"role": "user", "content": query}])
    
    def simulate_patient(self, scenario: Dict[str, Any], student_question: str) -> str:
        prompt = f"""Simulate a patient for medical student interview practice.
Patient: {scenario.get('demographics', 'Adult patient')}
Chief Complaint: {scenario.get('chief_complaint', 'Unknown')}
History: {scenario.get('history', 'None')}
Symptoms: {scenario.get('symptoms', 'Unknown')}

Respond naturally as this patient. Keep responses concise but realistic.
Student asks: {student_question}"""
        return self.provider.generate([
            {"role": "system", "content": "You are a simulated patient for medical education."},
            {"role": "user", "content": prompt}
        ])
    
    def generate_feedback(self, student_questions: List[str], scenario: Dict[str, Any]) -> str:
        prompt = f"""Analyze this medical student's interview performance.
Scenario: {scenario.get('name', 'Clinical Interview')}
Required Concepts: {scenario.get('required_concepts', [])}

Student's Questions:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(student_questions))}

Provide constructive feedback on coverage, technique, and missed areas."""
        return self.provider.generate([{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}])


class LLMClient:
    """OpenRouter-based LLM client for CLARA medical dialogue analysis.

    Uses Deepseek R1 for reasoning tasks and a smaller model for structured
    analysis (concept extraction, classification). Reads OPENROUTER_API_KEY
    from the environment.
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    model = "deepseek/deepseek-r1-0528"
    STRUCTURED_MODEL = "qwen/qwen3-8b"

    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _call(self, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 1024) -> str:
        import requests as _requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        resp = _requests.post(self.OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def analyze_medical_utterance(self, text: str) -> Dict[str, Any]:
        """Classify a student utterance and extract medical concepts.

        Returns dict with: concepts (list), question_type (open/closed/statement/unclear),
        clinical_relevance (high/medium/low).
        """
        prompt = (
            "Analyze the following medical student utterance. "
            "Return a JSON object with exactly these keys:\n"
            '- "concepts": list of medical concepts mentioned (strings)\n'
            '- "question_type": one of "open", "closed", "statement", "unclear"\n'
            '- "clinical_relevance": one of "high", "medium", "low"\n\n'
            f'Utterance: "{text}"\n\n'
            "Respond with only the JSON object, no extra text."
        )
        messages = [
            {"role": "system", "content": "You are a clinical education analysis assistant. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self._call(messages, model=self.STRUCTURED_MODEL)
            import json as _json
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return _json.loads(match.group())
        except Exception:
            pass
        return {"concepts": [], "question_type": "unclear", "clinical_relevance": "medium"}

    def generate_feedback(
        self,
        student_concepts: List[str],
        missing_concepts: List[str],
        communication_notes: str = "",
        scenario: str = "chest_pain_diagnosis",
    ) -> str:
        """Generate constructive clinical education feedback."""
        prompt = (
            f"Scenario: {scenario}\n"
            f"Concepts the student covered: {', '.join(student_concepts) or 'none'}\n"
            f"Concepts the student missed: {', '.join(missing_concepts) or 'none'}\n"
            f"Communication observations: {communication_notes or 'none'}\n\n"
            "Provide concise, constructive feedback for this medical student. "
            "Focus on what they did well, what was missing, and specific suggestions to improve."
        )
        messages = [
            {"role": "system", "content": "You are a clinical medical education expert providing feedback to trainees."},
            {"role": "user", "content": prompt},
        ]
        return self._call(messages)

    def generate_case_recommendation(
        self,
        student_gaps: List[str],
        student_id: str = "",
        recent_performance: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Recommend the next case and learning focus based on student gaps."""
        perf = recent_performance or {}
        prompt = (
            f"Student: {student_id or 'unknown'}\n"
            f"Knowledge/skill gaps: {', '.join(student_gaps) or 'none identified'}\n"
            f"Recent performance metrics: {perf}\n\n"
            "Based on these gaps and performance, recommend:\n"
            "1. The next clinical case type to practice\n"
            "2. Specific learning focus areas\n"
            "3. One concrete study tip\n"
            "Keep the response brief and actionable."
        )
        messages = [
            {"role": "system", "content": "You are an adaptive medical education advisor."},
            {"role": "user", "content": prompt},
        ]
        return self._call(messages)


def get_llm_response(prompt: str, provider: str = "groq") -> str:
    """Quick function to get an LLM response."""
    return MedicalLLM(LLMConfig(provider=provider)).medical_query(prompt)


if __name__ == "__main__":
    config = LLMConfig()
    llm = MedicalLLM(config)
    print(f"Testing {config.provider}...")
    if llm.is_available():
        print("✅ Connected!")
        print(llm.medical_query("What are symptoms of MI? Brief answer.")[:300])
    else:
        print(f"❌ Set {config.provider.upper()}_API_KEY")
