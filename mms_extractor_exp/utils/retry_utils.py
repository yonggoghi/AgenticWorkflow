"""
Retry Utilities - Retry and fallback logic for LLM calls
========================================================

This module provides retry decorators and configuration for handling
transient LLM API failures with automatic retry and fallback model support.
"""

import time
import logging
from functools import wraps
from typing import Callable, List, Type, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior
    
    Attributes:
        max_attempts: Maximum number of retry attempts (including initial attempt)
        retry_delay_seconds: Delay between retry attempts in seconds
        fallback_models: List of fallback model names to try if primary fails
        retry_exceptions: Exception types that should trigger a retry
        exponential_backoff: Whether to use exponential backoff for delays
    """
    max_attempts: int = 3
    retry_delay_seconds: float = 1.0
    fallback_models: List[str] = field(default_factory=lambda: ['ax', 'gpt'])
    retry_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [Exception]  # Retry on any exception by default
    )
    exponential_backoff: bool = False
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if self.exponential_backoff:
            return self.retry_delay_seconds * (2 ** (attempt - 1))
        return self.retry_delay_seconds


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Decorator to add retry logic to LLM calls
    
    Args:
        config: RetryConfig instance with retry settings
        on_retry: Optional callback function called on each retry
                  Signature: on_retry(exception, attempt_number)
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @with_retry(RetryConfig(max_attempts=3, retry_delay_seconds=1.0))
        def call_llm(prompt):
            return llm.invoke(prompt)
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    logger.debug(f"Attempt {attempt}/{config.max_attempts} for {func.__name__}")
                    result = func(*args, **kwargs)
                    
                    # Success on first attempt
                    if attempt == 1:
                        logger.debug(f"{func.__name__} succeeded on first attempt")
                    else:
                        logger.info(f"✅ {func.__name__} succeeded on attempt {attempt}")
                    
                    return result
                    
                except tuple(config.retry_exceptions) as e:
                    last_exception = e
                    
                    # Log the retry
                    if attempt < config.max_attempts:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"⚠️ {func.__name__} failed on attempt {attempt}/{config.max_attempts}: {e}"
                        )
                        logger.info(f"🔄 Retrying in {delay}s...")
                        
                        # Call retry callback if provided
                        if on_retry:
                            try:
                                on_retry(e, attempt)
                            except Exception as callback_error:
                                logger.error(f"Retry callback failed: {callback_error}")
                        
                        # Wait before retry
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"❌ {func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
            
            # All retries exhausted
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


class LLMRetryManager:
    """
    Manager for LLM calls with retry and fallback model support
    
    This class handles the complexity of trying multiple models
    when the primary model fails.
    """
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """
        Initialize retry manager
        
        Args:
            retry_config: Configuration for retry behavior
        """
        self.config = retry_config or RetryConfig()
    
    def call_with_fallback(
        self,
        primary_call: Callable[[], Any],
        fallback_calls: Optional[List[Callable[[], Any]]] = None,
        fallback_models: Optional[List[str]] = None
    ) -> Any:
        """
        Call primary function with fallback to other models
        
        Args:
            primary_call: Primary function to call
            fallback_calls: List of fallback functions to try
            fallback_models: List of model names for logging (optional)
        
        Returns:
            Result from successful call
        
        Raises:
            Exception: If all attempts (primary + fallbacks) fail
        """
        if fallback_calls is None:
            fallback_calls = []
        
        if fallback_models is None:
            fallback_models = [f"fallback_{i}" for i in range(len(fallback_calls))]
        
        # Try primary call with retry
        logger.info("🎯 Calling primary model...")
        try:
            @with_retry(self.config)
            def retried_primary():
                return primary_call()
            
            return retried_primary()
            
        except Exception as primary_error:
            logger.error(f"❌ Primary model failed: {primary_error}")
            
            # Try fallback models
            for i, (fallback_call, model_name) in enumerate(zip(fallback_calls, fallback_models)):
                logger.info(f"🔄 Trying fallback model {i+1}/{len(fallback_calls)}: {model_name}")
                
                try:
                    @with_retry(self.config)
                    def retried_fallback():
                        return fallback_call()
                    
                    result = retried_fallback()
                    logger.info(f"✅ Fallback model '{model_name}' succeeded!")
                    return result
                    
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback model '{model_name}' failed: {fallback_error}")
                    if i == len(fallback_calls) - 1:
                        # Last fallback also failed
                        logger.error("❌ All models failed (primary + all fallbacks)")
                        raise
            
            # No fallbacks were provided, re-raise primary error
            raise primary_error
