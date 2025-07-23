"""
Language model initialization and management.
"""
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from datetime import datetime
import logging

from ..config.settings import API_CONFIG, MODEL_CONFIG, get_device


logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages sentence embeddings and similarity calculations."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name or MODEL_CONFIG.embedding_model
        self.device = device or get_device()
        self.model = None
        
        logger.info(f"Initializing embedding model: {self.model_name} on {self.device}")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            # Check model loading mode and use local path if available
            if MODEL_CONFIG.model_loading_mode in ['local', 'auto']:
                import os
                local_path = MODEL_CONFIG.local_embedding_model_path
                
                # Check if local model exists
                if os.path.exists(local_path):
                    logger.info(f"Loading model from local path: {local_path}")
                    self.model = SentenceTransformer(local_path).to(self.device)
                    logger.info(f"Successfully loaded embedding model from local path on {self.device}")
                    return
                elif MODEL_CONFIG.model_loading_mode == 'local':
                    # If local mode is explicitly set but model doesn't exist, raise error
                    raise FileNotFoundError(f"Local model not found at {local_path}. Set MODEL_LOADING_MODE to 'auto' or 'remote' to download from internet.")
                else:
                    # Auto mode: fall back to remote if local doesn't exist
                    logger.warning(f"Local model not found at {local_path}, falling back to remote download")
            
            # Load from remote (Hugging Face)
            logger.info(f"Loading model from Hugging Face: {self.model_name}")
            self.model = SentenceTransformer(self.model_name).to(self.device)
            logger.info(f"Successfully loaded embedding model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 64, 
               convert_to_tensor: bool = True, show_progress_bar: bool = True) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            convert_to_tensor: Whether to return as tensor
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Encoded embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=show_progress_bar
        )
    
    def save_embeddings(self, embeddings: torch.Tensor, texts: List[str], filename: str):
        """
        Save embeddings to file.
        
        Args:
            embeddings: Embeddings tensor
            texts: Corresponding texts
            filename: Output filename
        """
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        
        np.savez_compressed(
            filename,
            embeddings=embeddings,
            texts=texts,
            timestamp=str(datetime.now())
        )
        logger.info(f"Saved embeddings to {filename}")
    
    def load_embeddings(self, filename: str) -> tuple:
        """
        Load embeddings from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (embeddings, texts)
        """
        data = np.load(filename, allow_pickle=True)
        embeddings = data['embeddings']
        texts = data['texts']
        timestamp = data['timestamp'] if 'timestamp' in data else None
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {filename}")
        if timestamp:
            logger.info(f"Created: {timestamp}")
        
        return embeddings, texts


class LLMManager:
    """Manages different language models for text generation."""
    
    def __init__(self):
        """Initialize LLM manager with multiple models."""
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all language models."""
        try:
            # Claude 3.7 Sonnet via SKT
            self.models['claude_37'] = ChatOpenAI(
                temperature=MODEL_CONFIG.temperature,
                openai_api_key=API_CONFIG.llm_api_key,
                openai_api_base=API_CONFIG.llm_api_url,
                model=MODEL_CONFIG.claude_model,
                max_tokens=MODEL_CONFIG.max_tokens
            )
            
            # Gemma 3 12B
            self.models['gemma_3'] = ChatOpenAI(
                temperature=MODEL_CONFIG.temperature,
                openai_api_key=API_CONFIG.llm_api_key,
                openai_api_base=API_CONFIG.llm_api_url,
                model=MODEL_CONFIG.gemma_model,
                max_tokens=MODEL_CONFIG.max_tokens
            )
            
            # GPT-4.1
            self.models['gpt_4'] = ChatOpenAI(
                temperature=MODEL_CONFIG.temperature,
                model=MODEL_CONFIG.gpt_model,
                openai_api_key=API_CONFIG.openai_api_key,
                max_tokens=2000
            )
            
            # Claude Sonnet 4
            self.models['claude_sonnet_4'] = ChatAnthropic(
                api_key=API_CONFIG.anthropic_api_key,
                model=MODEL_CONFIG.claude_sonnet_model,
                max_tokens=MODEL_CONFIG.max_tokens
            )
            
            # OpenAI Client for direct API calls
            self.openai_client = OpenAI(
                api_key=API_CONFIG.llm_api_key,
                base_url=API_CONFIG.llm_api_url
            )
            
            logger.info("Successfully initialized all language models")
            
        except Exception as e:
            logger.error(f"Failed to initialize language models: {e}")
            raise
    
    def get_model(self, model_name: str):
        """
        Get a specific model.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Language model instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.models.keys())}")
        
        return self.models[model_name]
    
    def generate(self, prompt: str, model_name: str = 'gemma_3') -> str:
        """
        Generate text using specified model.
        
        Args:
            prompt: Input prompt
            model_name: Name of the model to use
            
        Returns:
            Generated text
        """
        model = self.get_model(model_name)
        try:
            response = model.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate text with {model_name}: {e}")
            raise
    
    def generate_with_openai_client(self, prompt: str, model: str = None, 
                                  system_message: str = "You are a helpful assistant.") -> str:
        """
        Generate text using OpenAI client directly.
        
        Args:
            prompt: Input prompt
            model: Model name to use
            system_message: System message for the conversation
            
        Returns:
            Generated text
        """
        model = model or MODEL_CONFIG.gemma_model
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=MODEL_CONFIG.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate text with OpenAI client: {e}")
            raise


class CustomOpenAI:
    """Custom OpenAI wrapper for compatibility."""
    
    def __init__(self, model: str = None):
        """
        Initialize custom OpenAI client.
        
        Args:
            model: Model name to use
        """
        self.model = model or MODEL_CONFIG.gemma_model
        self.client = OpenAI(
            api_key=API_CONFIG.llm_api_key,
            base_url=API_CONFIG.llm_api_url
        )
    
    def __call__(self, prompt: str) -> str:
        """
        Generate text for the given prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content


def get_relevant_context(query: str, vectorstore: Any, topk: int = 5) -> Dict[str, str]:
    """
    Get relevant context from vectorstore.
    
    Args:
        query: Search query
        vectorstore: Vector store instance
        topk: Number of top results to return
        
    Returns:
        Dictionary with titles and context
    """
    try:
        docs = vectorstore.similarity_search(query, k=topk)
        context = "\n\n".join([doc.page_content for doc in docs])
        titles = ", ".join(set([
            doc.metadata['title'] 
            for doc in docs 
            if 'title' in doc.metadata.keys()
        ]))
        return {'title': titles, 'context': context}
    except Exception as e:
        logger.error(f"Failed to get relevant context: {e}")
        return {'title': '', 'context': ''}


def answer_question(query: str, vectorstore: Any, model_name: str = None) -> str:
    """
    Answer question using RAG approach.
    
    Args:
        query: Input query
        vectorstore: Vector store instance
        model_name: Model name to use
        
    Returns:
        Generated answer
    """
    # Get relevant context
    context = get_relevant_context(query, vectorstore)
    
    # Create combined prompt
    prompt = f"""Answer the following question based on the provided context:

Context: {context}

Question: {query}

Answer:"""
    
    # Use OpenAI directly
    custom_llm = CustomOpenAI(model_name)
    response = custom_llm(prompt)
    
    return response


def check_mps_availability() -> bool:
    """
    Check if MPS is available on this Mac.
    
    Returns:
        True if MPS is available, False otherwise
    """
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        logger.info("✅ MPS is available and ready to use!")
        return True
    else:
        logger.info("❌ MPS is not available. Using CPU instead.")
        return False 