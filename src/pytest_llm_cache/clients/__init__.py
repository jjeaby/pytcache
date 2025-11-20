"""Helper clients bundled with pytest-llm-cache for testing/demo purposes."""

from .gemini_client import call_llm as gemini_call
from .openai_client import call_openai as openai_call

__all__ = ["gemini_call", "openai_call"]
