"""
Entity extraction module using Kiwi for Korean text processing.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from kiwipiepy import Kiwi
import pandas as pd
import logging

from ..utils.text_processing import preprocess_text, filter_specific_terms
from ..utils.similarity import parallel_fuzzy_similarity, parallel_seq_similarity
from ..config.settings import EXTRACTION_SCHEMA, PROCESSING_CONFIG


logger = logging.getLogger(__name__)


class Token:
    """Token class for compatibility."""
    
    def __init__(self, form: str, tag: str, start: int, length: int):
        self.form = form
        self.tag = tag
        self.start = start
        self.len = length


class Sentence:
    """Sentence class for compatibility."""
    
    def __init__(self, text: str, start: int, end: int, tokens: List[Token], subs: List = None):
        self.text = text
        self.start = start
        self.end = end
        self.tokens = tokens
        self.subs = subs or []


class KiwiEntityExtractor:
    """Entity extractor using Kiwi Korean morphological analyzer."""
    
    def __init__(self, entity_vocab: List[Tuple[str, Dict[str, Any]]], 
                 stop_words: List[str] = None):
        """
        Initialize Kiwi entity extractor.
        
        Args:
            entity_vocab: List of (entity_name, metadata) tuples
            stop_words: List of stop words to exclude
        """
        self.entity_vocab = entity_vocab
        self.stop_words = stop_words or []
        self.kiwi = Kiwi()
        self.exclusion_patterns = EXTRACTION_SCHEMA.exclusion_tag_patterns
        
        logger.info("Initializing Kiwi entity extractor")
        self._initialize_kiwi()
    
    def _initialize_kiwi(self):
        """Initialize Kiwi with user-defined entities."""
        entity_list = list(set([item[0] for item in self.entity_vocab]))
        
        for word in entity_list:
            if word not in self.stop_words:
                try:
                    self.kiwi.add_user_word(word, "NNP")
                except Exception as e:
                    logger.warning(f"Failed to add word '{word}' to Kiwi: {e}")
        
        logger.info(f"Added {len(entity_list)} entities to Kiwi vocabulary")
    
    def filter_text_by_exclusion_patterns(self, sentence: Sentence, 
                                        exclusion_patterns: List[List[str]]) -> str:
        """
        Filter text by replacing tokens that match exclusion patterns with whitespace.
        
        Args:
            sentence: Sentence object with tokens
            exclusion_patterns: List of exclusion tag patterns
            
        Returns:
            Filtered text
        """
        # Separate individual tags from sequences
        individual_tags = set()
        sequences = []
        
        for pattern in exclusion_patterns:
            if isinstance(pattern, list):
                if len(pattern) == 1:
                    individual_tags.add(pattern[0])
                else:
                    sequences.append(pattern)
            else:
                individual_tags.add(pattern)
        
        # Track which tokens to exclude
        tokens_to_exclude = set()
        
        # Check for individual tag matches
        for i, token in enumerate(sentence.tokens):
            if token.tag in individual_tags:
                tokens_to_exclude.add(i)
        
        # Check for sequence matches
        for sequence in sequences:
            seq_len = len(sequence)
            for i in range(len(sentence.tokens) - seq_len + 1):
                if all(sentence.tokens[i + j].tag == sequence[j] for j in range(seq_len)):
                    for j in range(seq_len):
                        tokens_to_exclude.add(i + j)
        
        # Create filtered text
        result_chars = list(sentence.text)
        
        for i, token in enumerate(sentence.tokens):
            if i in tokens_to_exclude:
                start_pos = token.start - sentence.start
                end_pos = start_pos + token.len
                for j in range(start_pos, end_pos):
                    if j < len(result_chars) and result_chars[j] != ' ':
                        result_chars[j] = ' '
        
        filtered_text = ''.join(result_chars)
        return re.sub(r'\s+', ' ', filtered_text)
    
    def extract_entities_from_text(self, text: str) -> List[str]:
        """
        Extract named entities from text using Kiwi.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        try:
            # Tokenize and extract entities
            result = self.kiwi.tokenize(text, normalize_coda=True, z_coda=False, split_complex=False)
            entities = []
            
            for token in result:
                if (token.tag == 'NNP' 
                    and token.form not in self.stop_words + ['-'] 
                    and len(token.form) >= 2 
                    and token.form.lower() not in [word.lower() for word in self.stop_words]):
                    entities.append(token.form)
            
            return filter_specific_terms(entities)
        except Exception as e:
            logger.warning(f"Failed to extract entities from text: {e}")
            return []
    
    def extract_sentences_from_text(self, text: str) -> List[str]:
        """
        Extract and filter sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of filtered sentences
        """
        try:
            # Split into sentences
            sentences = sum(self.kiwi.split_into_sents(
                re.split(r"_+",text), 
                return_tokens=True, 
                return_sub_sents=True
            ), [])
            
            # Process all sentences including sub-sentences
            sentences_all = []
            for sent in sentences:
                if hasattr(sent, 'subs') and sent.subs:
                    for sub_sent in sent.subs:
                        sentences_all.append(sub_sent)
                else:
                    sentences_all.append(sent)
            
            # Filter sentences using exclusion patterns
            sentence_list = []
            for sent in sentences_all:
                try:
                    filtered_text = self.filter_text_by_exclusion_patterns(sent, self.exclusion_patterns)
                    sentence_list.append(filtered_text)
                except Exception as e:
                    # If filtering fails, use original text
                    sentence_list.append(sent.text)
            
            return sentence_list
        except Exception as e:
            logger.warning(f"Failed to extract sentences from text: {e}")
            # Return simple split as fallback
            return [text.strip()]
    
    def extract_entities_from_message(self, message: str, item_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Extract entities from MMS message.
        
        Args:
            message: MMS message text
            item_df: DataFrame containing item information
            
        Returns:
            Tuple of (candidate_items, extra_item_df)
        """
        logger.info("Extracting entities from message")
        
        try:
            # Extract sentences and entities
            sentence_list = self.extract_sentences_from_text(message)
            entities_from_kiwi = self.extract_entities_from_text(message)
            
            logger.info(f"Extracted {len(entities_from_kiwi)} entities from Kiwi")
            
            # Get unique item aliases, ensuring they're strings
            item_aliases = [str(alias) for alias in item_df['item_nm_alias'].unique() if pd.notna(alias)]
            
            # Calculate similarities using fuzzy matching
            similarities_fuzzy = parallel_fuzzy_similarity(
                sentence_list, 
                item_aliases, 
                threshold=PROCESSING_CONFIG.fuzzy_threshold,
                text_col_nm='sent',
                item_col_nm='item_nm_alias',
                n_jobs=PROCESSING_CONFIG.n_jobs,
                batch_size=30
            )
            
            # Calculate sequence similarities if we have fuzzy results
            if not similarities_fuzzy.empty:
                similarities_seq = parallel_seq_similarity(
                    sent_item_pdf=similarities_fuzzy,
                    text_col_nm='sent',
                    item_col_nm='item_nm_alias',
                    n_jobs=PROCESSING_CONFIG.n_jobs,
                    batch_size=PROCESSING_CONFIG.batch_size
                )
            else:
                similarities_seq = pd.DataFrame(columns=['sent', 'item_nm_alias', 'sim'])
            
            # Filter candidate items using safe query operations
            if not similarities_seq.empty:
                # Use safe filtering instead of query
                threshold_mask = similarities_seq['sim'] >= PROCESSING_CONFIG.similarity_threshold
                non_empty_mask = similarities_seq['item_nm_alias'].str.len() > 0
                stop_words_mask = ~similarities_seq['item_nm_alias'].isin(self.stop_words)
                
                cand_items = similarities_seq[threshold_mask & non_empty_mask & stop_words_mask]
            else:
                cand_items = pd.DataFrame(columns=['sent', 'item_nm_alias', 'sim'])
            
            # Process entities from Kiwi
            entities_from_kiwi_df = pd.DataFrame()
            if entities_from_kiwi:
                # Safe filtering for Kiwi entities
                kiwi_mask = item_df['item_nm_alias'].isin(entities_from_kiwi)
                entities_from_kiwi_df = item_df[kiwi_mask][['item_nm', 'item_nm_alias']].copy()
                entities_from_kiwi_df['sim'] = 1.0
            
            # Combine results
            if not cand_items.empty or not entities_from_kiwi_df.empty:
                # Ensure column compatibility
                if not cand_items.empty:
                    cand_items_clean = cand_items[['item_nm_alias', 'sim']].copy()
                else:
                    cand_items_clean = pd.DataFrame(columns=['item_nm_alias', 'sim'])
                
                if not entities_from_kiwi_df.empty:
                    kiwi_clean = entities_from_kiwi_df[['item_nm_alias', 'sim']].copy()
                else:
                    kiwi_clean = pd.DataFrame(columns=['item_nm_alias', 'sim'])
                
                # Combine DataFrames
                if not cand_items_clean.empty and not kiwi_clean.empty:
                    cand_item_df = pd.concat([cand_items_clean, kiwi_clean], ignore_index=True)
                elif not cand_items_clean.empty:
                    cand_item_df = cand_items_clean
                elif not kiwi_clean.empty:
                    cand_item_df = kiwi_clean
                else:
                    cand_item_df = pd.DataFrame(columns=['item_nm_alias', 'sim'])
                
                # Get candidate item list
                if not cand_item_df.empty:
                    cand_item_list = (cand_item_df
                                     .sort_values('sim', ascending=False)
                                     .groupby(["item_nm_alias"])['sim']
                                     .max()
                                     .reset_index(name='final_sim')
                                     .sort_values('final_sim', ascending=False))
                    
                    # Filter by final similarity threshold
                    cand_item_list = cand_item_list[cand_item_list['final_sim'] >= 0.2]['item_nm_alias'].unique()
                else:
                    cand_item_list = []
            else:
                cand_item_list = []
            
            # Create extra item DataFrame
            if len(cand_item_list) > 0:
                # Safe filtering for final results
                final_mask = item_df['item_nm_alias'].isin(cand_item_list)
                extra_item_df = (item_df[final_mask]
                                [['item_nm', 'item_nm_alias', 'item_id']]
                                .groupby(["item_nm"])['item_id']
                                .apply(list)
                                .reset_index())
            else:
                extra_item_df = pd.DataFrame(columns=['item_nm', 'item_id'])
            
            logger.info(f"Found {len(cand_item_list)} candidate items")
            
            return list(cand_item_list), extra_item_df
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            # Return empty results on error
            return [], pd.DataFrame(columns=['item_nm', 'item_id'])


def create_entity_extractor(entity_vocab: List[Tuple[str, Dict[str, Any]]], 
                          stop_words: List[str] = None) -> KiwiEntityExtractor:
    """
    Factory function to create entity extractor.
    
    Args:
        entity_vocab: List of (entity_name, metadata) tuples
        stop_words: List of stop words to exclude
        
    Returns:
        Initialized KiwiEntityExtractor
    """
    return KiwiEntityExtractor(entity_vocab, stop_words) 