# LLM Integration for CLARA

This adds AI-powered patient simulation and feedback to CLARA using Groq and HuggingFace APIs.

## 📦 Files Included

**`src/clara/llm.py`**
- Main LLM integration module
- Connects to Groq (fast, free) and HuggingFace (medical-specialized)
- Provides 3 core functions:
  - `medical_query()` - Ask medical questions
  - `simulate_patient()` - Generate realistic patient responses
  - `generate_feedback()` - Analyze student interview performance

**`scripts/view_full_responses.py`**
- Compare full responses from both Groq and HuggingFace side-by-side
- Shows response length, speed, and quality differences
- Useful for choosing which model to use

**`requirements.txt`**
- Lists all Python packages needed (groq, huggingface_hub, python-dotenv)

**`.env.example`**
- Template showing what API keys are needed
- Copy this to `.env` and add your actual keys

## 🚀 Quick Start

### 1. Install Packages
```bash
pip install groq python-dotenv huggingface_hub
```

### 2. Get API Keys (Both Free!)
- **Groq**: https://console.groq.com/keys
- **HuggingFace**: https://huggingface.co/settings/tokens

### 3. Configure
```bash
cp .env.example .env
# Edit .env and paste your API keys
```

### 4. Test It
```bash
# Test connection
python -c "from src.clara.llm import MedicalLLM, LLMConfig; llm = MedicalLLM(LLMConfig(provider='groq')); print(llm.medical_query('What are MI symptoms?'))"

# Compare both models
cd scripts
python view_full_responses.py
```

##  How to Use in Code

```python
from clara.llm import MedicalLLM, LLMConfig

# Initialize (choose 'groq' or 'huggingface')
llm = MedicalLLM(LLMConfig(provider="groq"))

# Simulate a patient
scenario = {
    "demographics": "58yo male",
    "chief_complaint": "chest pain",
    "symptoms": "Crushing chest pain, left arm radiation"
}
response = llm.simulate_patient(scenario, "Describe your pain")
print(response)

# Get feedback on interview
questions = ["What brought you in?", "When did it start?"]
feedback = llm.generate_feedback(questions, {"name": "Chest Pain"})
print(feedback)
```

##  Model Comparison

| Model | Speed | Best For |
|-------|-------|----------|
| **Groq** | Fast (2s) | Real-time patient simulation |
| **HuggingFace** | Slow (35s) | Detailed medical reasoning |


