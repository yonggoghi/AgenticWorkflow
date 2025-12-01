"""
LLM utility for mms_agent
Reuses configuration from mms_extractor_exp
"""

import os
import sys
from typing import Optional
from langchain_openai import ChatOpenAI

# Add mms_extractor_exp to path
mms_exp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mms_extractor_exp")
if mms_exp_path not in sys.path:
    sys.path.insert(0, mms_exp_path)

from config.settings import API_CONFIG, MODEL_CONFIG


class LLMClient:
    """
    Singleton LLM client using mms_extractor_exp configuration
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.llm_model = self._create_llm()
        self._initialized = True
    
    def _create_llm(self) -> ChatOpenAI:
        """Create LLM using existing config"""
        # Use config from mms_extractor_exp
        model_kwargs = {
            "model": MODEL_CONFIG.llm_model,
            "api_key": API_CONFIG.llm_api_key,
            "base_url": API_CONFIG.llm_api_url,
            "max_tokens": MODEL_CONFIG.llm_max_tokens,
            "temperature": MODEL_CONFIG.temperature,
        }
        
        return ChatOpenAI(**model_kwargs)
    
    def invoke(self, prompt: str) -> str:
        """Simple LLM invocation"""
        response = self.llm_model.invoke(prompt)
        return response.content


# Singleton instance
_llm_client = LLMClient()


def get_llm() -> LLMClient:
    """Get the singleton LLM client"""
    return _llm_client
