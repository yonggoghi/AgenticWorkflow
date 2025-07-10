"""
LLM processing module for handling language model interactions.
"""

from typing import List, Dict, Any, Optional, Union
import json
import time
from dataclasses import dataclass
import openai
import anthropic
from ..config.settings import (
    API_CONFIG,
    LLM_PROCESSING_CONFIG,
    SCHEMA_DEFINITIONS
)

@dataclass
class LLMResponse:
    """Container for LLM response data."""
    content: str
    model: str
    usage: Dict[str, int]
    latency: float
    error: Optional[str] = None

class LLMProcessor:
    """
    A class for processing text using language models.
    
    This class handles interactions with various LLM providers
    and processes text according to specified schemas.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_model: str = LLM_PROCESSING_CONFIG["default_model"],
        temperature: float = LLM_PROCESSING_CONFIG["temperature"],
        max_tokens: int = LLM_PROCESSING_CONFIG["max_tokens"],
        retry_count: int = LLM_PROCESSING_CONFIG["retry_count"],
        retry_delay: float = LLM_PROCESSING_CONFIG["retry_delay"]
    ):
        """
        Initialize the LLMProcessor.
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            default_model: Default model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        # Set API keys
        self.openai_api_key = openai_api_key or API_CONFIG["openai_api_key"]
        self.anthropic_api_key = anthropic_api_key or API_CONFIG["anthropic_api_key"]
        
        # Initialize clients
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        if self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.anthropic_api_key
            )
        
        # Set processing parameters
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        # Load schema definitions
        self.schemas = SCHEMA_DEFINITIONS
    
    def process_text(
        self,
        text: str,
        schema_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Process text using LLM according to specified schema.
        
        Args:
            text: Input text to process
            schema_name: Name of schema to use
            model: Model to use (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens (overrides default)
            system_prompt: Custom system prompt (overrides schema)
            
        Returns:
            LLMResponse object containing processed text and metadata
        """
        if not text:
            return LLMResponse(
                content="",
                model=model or self.default_model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency=0.0,
                error="Empty input text"
            )
        
        # Get schema
        schema = self.schemas.get(schema_name)
        if not schema:
            return LLMResponse(
                content="",
                model=model or self.default_model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency=0.0,
                error=f"Unknown schema: {schema_name}"
            )
        
        # Prepare parameters
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        system_prompt = system_prompt or schema.get("system_prompt", "")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Process with retries
        for attempt in range(self.retry_count + 1):
            try:
                start_time = time.time()
                
                if model.startswith("gpt-"):
                    response = self._process_openai(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                elif model.startswith("claude-"):
                    response = self._process_anthropic(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    return LLMResponse(
                        content="",
                        model=model,
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        latency=0.0,
                        error=f"Unsupported model: {model}"
                    )
                
                latency = time.time() - start_time
                
                return LLMResponse(
                    content=response["content"],
                    model=model,
                    usage=response["usage"],
                    latency=latency
                )
                
            except Exception as e:
                if attempt == self.retry_count:
                    return LLMResponse(
                        content="",
                        model=model,
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        latency=0.0,
                        error=str(e)
                    )
                time.sleep(self.retry_delay)
    
    def _process_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Process text using OpenAI API."""
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    def _process_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Process text using Anthropic API."""
        # Convert messages to Anthropic format
        system = next(
            (msg["content"] for msg in messages if msg["role"] == "system"),
            ""
        )
        user_messages = [
            msg["content"] for msg in messages
            if msg["role"] == "user"
        ]
        
        response = self.anthropic_client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": "\n".join(user_messages)}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "content": response.content[0].text,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }
    
    def extract_structured_data(
        self,
        text: str,
        schema_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using LLM.
        
        Args:
            text: Input text
            schema_name: Name of schema to use
            model: Model to use (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens (overrides default)
            
        Returns:
            Extracted data as dictionary
        """
        # Get schema
        schema = self.schemas.get(schema_name)
        if not schema:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        # Prepare system prompt for structured extraction
        system_prompt = f"""You are a precise data extraction assistant.
Extract information from the input text according to the following schema:
{json.dumps(schema['fields'], indent=2, ensure_ascii=False)}

Rules:
1. Return ONLY valid JSON matching the schema
2. Use null for missing values
3. Preserve original text case
4. Include confidence scores (0-1) for each field
5. Explain any assumptions made in the 'notes' field

Example output format:
{{
    "field1": {{
        "value": "extracted value",
        "confidence": 0.95,
        "source": "relevant text snippet"
    }},
    "notes": "Assumptions and explanations"
}}"""
        
        # Process text
        response = self.process_text(
            text=text,
            schema_name=schema_name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
        
        if response.error:
            raise ValueError(f"LLM processing failed: {response.error}")
        
        try:
            # Parse JSON response
            data = json.loads(response.content)
            
            # Validate against schema
            self._validate_extracted_data(data, schema["fields"])
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")
    
    def _validate_extracted_data(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> None:
        """Validate extracted data against schema."""
        # Check required fields
        for field, field_schema in schema.items():
            if field_schema.get("required", False):
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
                
                # Check field structure
                if not isinstance(data[field], dict):
                    raise ValueError(f"Invalid structure for field {field}")
                
                if "value" not in data[field]:
                    raise ValueError(f"Missing 'value' in field {field}")
                
                # Validate value type
                expected_type = field_schema.get("type", "string")
                if expected_type == "string" and not isinstance(data[field]["value"], str):
                    raise ValueError(f"Invalid type for field {field}")
                elif expected_type == "number" and not isinstance(data[field]["value"], (int, float)):
                    raise ValueError(f"Invalid type for field {field}")
                elif expected_type == "boolean" and not isinstance(data[field]["value"], bool):
                    raise ValueError(f"Invalid type for field {field}")
                elif expected_type == "array" and not isinstance(data[field]["value"], list):
                    raise ValueError(f"Invalid type for field {field}")
        
        # Check for unexpected fields
        for field in data:
            if field not in schema and field != "notes":
                raise ValueError(f"Unexpected field: {field}")
    
    def batch_process(
        self,
        texts: List[str],
        schema_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        batch_size: int = LLM_PROCESSING_CONFIG["batch_size"],
        max_concurrent: int = LLM_PROCESSING_CONFIG["max_concurrent"]
    ) -> List[Union[Dict[str, Any], LLMResponse]]:
        """
        Process multiple texts in batches.
        
        Args:
            texts: List of input texts
            schema_name: Name of schema to use
            model: Model to use (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens (overrides default)
            batch_size: Number of texts to process in each batch
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of processed results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for text in batch:
                try:
                    result = self.extract_structured_data(
                        text=text,
                        schema_name=schema_name,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    batch_results.append(result)
                except Exception as e:
                    batch_results.append(LLMResponse(
                        content="",
                        model=model or self.default_model,
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        latency=0.0,
                        error=str(e)
                    ))
            
            results.extend(batch_results)
            
            # Rate limiting
            if i + batch_size < len(texts):
                time.sleep(self.retry_delay)
        
        return results 