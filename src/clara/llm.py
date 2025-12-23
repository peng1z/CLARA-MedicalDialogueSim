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
