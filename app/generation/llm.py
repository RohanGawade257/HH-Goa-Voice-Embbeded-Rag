"""
HH-Goa-Rag — Generation sub-package
LLM integration placeholder.

This module is reserved for the next development phase:
  - Multilingual LLM client (e.g., Sarvam, OpenAI, Gemini)
  - Generative answer synthesis over retrieved context
  - Prompt templates for Hindi and other target languages

Current implementation uses extractive answers only (app/answer_generator.py).
"""

# Target languages (BCP-47 codes):
TARGET_LANGUAGES = [
    "as", "bn", "gu", "hi", "kn",
    "ml", "mr", "ne", "or", "pa",
    "sa", "ta", "te", "ur",
]
