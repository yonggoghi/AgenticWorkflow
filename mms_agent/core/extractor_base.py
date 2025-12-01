"""
Base extractor with essential functionality
Independent implementation without mms_extractor_exp dependencies
"""

import os
import sys
import logging
import pandas as pd
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ExtractorBase:
    """
    Base extractor with essential NLP and matching functionality
    Singleton pattern for efficiency
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
        
        # Import data loader
        from .data_loader import DataLoader
        
        # Load data
        self.data_loader = DataLoader().load_all()
        self.item_pdf = self.data_loader.item_pdf
        self.org_pdf = self.data_loader.org_pdf
        self.alias_pdf = self.data_loader.alias_pdf
        self.pgm_pdf = self.data_loader.pgm_pdf
        
        # Initialize NLP components
        self.kiwi = None
        self.emb_model = None
        self.clue_embeddings = None
        
        self._load_kiwi()
        self._load_embeddings()
        
        # Stop words
        self.stop_item_names = ['SKT', 'SK', 'T', '링크', '바로가기', '확인']
        
        self._initialized = True
    
    def _load_kiwi(self):
        """Load Kiwi morphological analyzer"""
        try:
            from kiwipiepy import Kiwi
            self.kiwi = Kiwi()
            logger.info("✅ Kiwi loaded successfully")
        except ImportError:
            logger.warning("⚠️  Kiwi not installed, morphological analysis unavailable")
            self.kiwi = None
        except Exception as e:
            logger.error(f"Failed to load Kiwi: {e}")
            self.kiwi = None
    
    def _load_embeddings(self):
        """Load embedding model for program classification"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Try to load model
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "mms_extractor_exp/models/jhgan-ko-sroberta-multitask"
            )
            
            if os.path.exists(model_path):
                self.emb_model = SentenceTransformer(model_path)
            else:
                # Fallback to default model
                self.emb_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # Compute clue embeddings if program data exists
            if not self.pgm_pdf.empty and 'clue_tag' in self.pgm_pdf.columns:
                clue_texts = self.pgm_pdf['clue_tag'].str.lower().tolist()
                self.clue_embeddings = self.emb_model.encode(
                    clue_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
            else:
                self.clue_embeddings = torch.tensor([])
            
            logger.info("✅ Embedding model loaded successfully")
        except ImportError:
            logger.warning("⚠️  SentenceTransformers not installed, classification unavailable")
            self.emb_model = None
            self.clue_embeddings = None
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            self.emb_model = None
            try:
                import torch
                self.clue_embeddings = torch.tensor([])
            except ImportError:
                self.clue_embeddings = None
    
    def extract_entities_from_kiwi(self, message: str) -> Tuple[List[str], pd.DataFrame]:
        """
        Extract entities using Kiwi morphological analysis
        Returns: (entities_list, matched_items_df)
        """
        if self.kiwi is None:
            logger.warning("Kiwi not available")
            return [], pd.DataFrame()
        
        if self.item_pdf.empty:
            logger.warning("Item data not loaded")
            return [], pd.DataFrame()
        
        try:
            # Tokenize
            tokens = self.kiwi.tokenize(message)
            
            # Extract nouns
            entities = []
            for token in tokens:
                if token.tag in ['NNG', 'NNP', 'SL']:  # Nouns and foreign words
                    if len(token.form) >= 2:
                        entities.append(token.form)
            
            # Remove duplicates
            entities = list(set(entities))
            
            # Match against item database
            matched_items = []
            for entity in entities:
                # Ensure columns are string type before using .str accessor
                matches = self.item_pdf[
                    self.item_pdf['item_nm_alias'].astype(str).str.contains(entity, case=False, na=False) |
                    self.item_pdf['item_nm'].astype(str).str.contains(entity, case=False, na=False)
                ]
                if not matches.empty:
                    matched_items.append(matches)
            
            if matched_items:
                extra_item_pdf = pd.concat(matched_items).drop_duplicates()
            else:
                extra_item_pdf = pd.DataFrame()
            
            return entities, extra_item_pdf
            
        except Exception as e:
            logger.error(f"Kiwi extraction failed: {e}")
            return [], pd.DataFrame()
