import os
from dotenv import load_dotenv
import anthropic
import openai

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")  # For sk-gapk-... keys
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    
    # Validate required keys
    @classmethod
    def validate_keys(cls):
        missing_keys = []
        if not cls.ANTHROPIC_API_KEY:
            missing_keys.append("ANTHROPIC_API_KEY")
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not cls.CUSTOM_API_KEY:
            missing_keys.append("CUSTOM_API_KEY")
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    # Helper methods for API clients
    @classmethod
    def get_anthropic_client(cls):
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return anthropic.Anthropic(api_key=cls.ANTHROPIC_API_KEY)
    
    @classmethod
    def get_openai_client(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return openai.OpenAI(api_key=cls.OPENAI_API_KEY)

# Create a global config instance
config = Config()

# Export commonly used values
ANTHROPIC_API_KEY = config.ANTHROPIC_API_KEY
OPENAI_API_KEY = config.OPENAI_API_KEY
CUSTOM_API_KEY = config.CUSTOM_API_KEY
HUGGINGFACE_TOKEN = config.HUGGINGFACE_TOKEN
