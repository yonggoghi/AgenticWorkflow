"""
Configuration file for MMS Extractor project.
This file manages API keys and other settings through environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for API keys and settings."""
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    
    # Model settings
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'claude-3-sonnet-20240229')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    
    # Paths
    DATA_PATH = os.getenv('DATA_PATH', './data')
    MODEL_PATH = os.getenv('MODEL_PATH', './models')
    
    # Other settings
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4000'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are present."""
        missing_keys = []
        
        if not cls.ANTHROPIC_API_KEY:
            missing_keys.append('ANTHROPIC_API_KEY')
        if not cls.OPENAI_API_KEY:
            missing_keys.append('OPENAI_API_KEY')
            
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    @classmethod
    def get_anthropic_client(cls):
        """Get Anthropic client with API key."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        try:
            from anthropic import Anthropic
            return Anthropic(api_key=cls.ANTHROPIC_API_KEY)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    @classmethod
    def get_openai_client(cls):
        """Get OpenAI client with API key."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        try:
            from openai import OpenAI
            return OpenAI(api_key=cls.OPENAI_API_KEY)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

# Global configuration instance
config = Config() 