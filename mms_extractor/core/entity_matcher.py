"""
Entity matching module for Korean text processing.
"""

from typing import List, Dict, Any, Tuple, Set, Optional
import re
from rapidfuzz import fuzz, process
import numpy as np
from ..config.settings import ENTITY_MATCHING_CONFIG

class KoreanEntityMatcher:
    """
    A class for fuzzy matching Korean entities in text.
    
    This class implements various similarity metrics and matching strategies
    optimized for Korean text processing.
    """
    
    def __init__(
        self,
        min_similarity: int = ENTITY_MATCHING_CONFIG["min_similarity"],
        ngram_size: int = ENTITY_MATCHING_CONFIG["ngram_size"],
        min_entity_length: int = ENTITY_MATCHING_CONFIG["min_entity_length"],
        token_similarity: bool = ENTITY_MATCHING_CONFIG["token_similarity"]
    ):
        """
        Initialize the KoreanEntityMatcher.
        
        Args:
            min_similarity: Minimum similarity score (0-100) for fuzzy matching
            ngram_size: Size of character n-grams to use for indexing
            min_entity_length: Minimum length of entities to consider
            token_similarity: Whether to use token-based similarity measures
        """
        self.min_similarity = min_similarity
        self.ngram_size = ngram_size
        self.min_entity_length = min_entity_length
        self.token_similarity = token_similarity
        self.entities: List[str] = []
        self.entity_data: Dict[str, Dict[str, Any]] = {}
        self.normalized_entities: Dict[str, str] = {}
        self.ngram_index: Dict[str, Set[str]] = {}
        
    def build_from_list(self, entities: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Build entity index from a list of entities.
        
        Args:
            entities: List of (entity_name, data) tuples
        """
        self.entities = []
        self.entity_data = {}
        
        for i, entity in enumerate(entities):
            if isinstance(entity, tuple) and len(entity) == 2:
                entity_name, data = entity
                self.entities.append(entity_name)
                self.entity_data[entity_name] = data
            else:
                self.entities.append(entity)
                self.entity_data[entity] = {'id': i, 'entity': entity}
                
        # Store normalized forms for faster lookup
        self.normalized_entities = {
            self._normalize_text(entity): entity 
            for entity in self.entities
        }
        
        # Build n-gram index
        self._build_ngram_index(n=self.ngram_size)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into Korean and non-Korean tokens."""
        return re.findall(r'[가-힣]+|[a-z0-9]+', self._normalize_text(text))
    
    def _build_ngram_index(self, n: int = 2) -> None:
        """Build n-gram index for faster candidate selection."""
        self.ngram_index = {}
        
        for entity in self.entities:
            if len(entity) < self.min_entity_length:
                continue
                
            normalized_entity = self._normalize_text(entity)
            entity_chars = list(normalized_entity)
            
            # Character n-grams
            for i in range(len(entity_chars) - n + 1):
                ngram = ''.join(entity_chars[i:i+n])
                if ngram not in self.ngram_index:
                    self.ngram_index[ngram] = set()
                self.ngram_index[ngram].add(entity)
            
            # Token-based n-grams
            if self.token_similarity:
                tokens = self._tokenize(normalized_entity)
                for token in tokens:
                    if len(token) >= n:
                        token_key = f"TOKEN:{token}"
                        if token_key not in self.ngram_index:
                            self.ngram_index[token_key] = set()
                        self.ngram_index[token_key].add(entity)
    
    def _get_candidates(self, text: str, n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get candidate entities based on n-gram overlap."""
        if n is None:
            n = self.ngram_size
            
        normalized_text = self._normalize_text(text)
        
        # Quick exact match check
        if normalized_text in self.normalized_entities:
            entity = self.normalized_entities[normalized_text]
            return [(entity, float('inf'))]
        
        # Generate n-grams for text
        text_chars = list(normalized_text)
        text_ngrams = set()
        
        # Character n-grams
        for i in range(len(text_chars) - n + 1):
            text_ngrams.add(''.join(text_chars[i:i+n]))
        
        # Token n-grams
        if self.token_similarity:
            tokens = self._tokenize(normalized_text)
            for token in tokens:
                if len(token) >= n:
                    text_ngrams.add(f"TOKEN:{token}")
        
        # Find candidates
        candidates = set()
        for ngram in text_ngrams:
            if ngram in self.ngram_index:
                candidates.update(self.ngram_index[ngram])
        
        # Score candidates
        candidate_scores = {}
        for candidate in candidates:
            score = self._calculate_similarity(text, candidate)
            if score >= self.min_similarity:
                candidate_scores[candidate] = score
        
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_similarity(self, text: str, entity: str) -> float:
        """Calculate similarity between text and entity using multiple metrics."""
        normalized_text = self._normalize_text(text)
        normalized_entity = self._normalize_text(entity)
        
        if normalized_text == normalized_entity:
            return 100.0
        
        # Basic string similarity
        ratio_score = fuzz.ratio(normalized_text, normalized_entity)
        
        # Partial string match
        partial_score = 0
        if normalized_text in normalized_entity:
            partial_score = (len(normalized_text) / len(normalized_entity)) * 100
        elif normalized_entity in normalized_text:
            partial_score = (len(normalized_entity) / len(normalized_text)) * 100
        
        # Token similarity
        token_score = 0
        if self.token_similarity:
            text_tokens = set(self._tokenize(normalized_text))
            entity_tokens = set(self._tokenize(normalized_entity))
            
            if text_tokens and entity_tokens:
                common_tokens = text_tokens.intersection(entity_tokens)
                all_tokens = text_tokens.union(entity_tokens)
                token_score = (len(common_tokens) / len(all_tokens)) * 100
        
        # Token order independent similarity
        token_sort_score = fuzz.token_sort_ratio(normalized_text, normalized_entity)
        token_set_score = fuzz.token_set_ratio(normalized_text, normalized_entity)
        
        # Weighted average of all metrics
        final_score = (
            ratio_score * 0.3 +
            max(partial_score, 0) * 0.1 +
            token_score * 0.2 +
            token_sort_score * 0.2 +
            token_set_score * 0.2
        )
        
        return final_score
    
    def find_entities(
        self,
        text: str,
        max_candidates_per_span: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find entity matches in text using fuzzy matching.
        
        Args:
            text: Text to search for entities
            max_candidates_per_span: Maximum number of candidates to consider per span
            
        Returns:
            List of matched entities with position and metadata
        """
        matches = []
        
        # Extract potential spans
        spans = self._extract_korean_spans(text)
        
        for span_text, start, end in spans:
            if len(span_text.strip()) < self.min_entity_length:
                continue
            
            # Get candidates
            candidates = self._get_candidates(span_text)
            if not candidates:
                continue
            
            # Process top candidates
            for entity, score in candidates[:max_candidates_per_span]:
                matches.append({
                    'text': span_text,
                    'matched_entity': entity,
                    'score': score,
                    'start': start,
                    'end': end,
                    'data': self.entity_data.get(entity, {})
                })
        
        # Sort by position and resolve overlaps
        matches.sort(key=lambda x: (x['start'], -x['score']))
        return self._resolve_overlapping_matches(matches)
    
    def _extract_korean_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract potential entity spans from text."""
        spans = []
        min_len = self.min_entity_length
        
        # Various patterns for Korean text
        patterns = [
            (r'[a-zA-Z]+[가-힣]+(?:\s+[가-힣가-힣a-zA-Z0-9]+)*', 0),  # Mixed Korean-English
            (r'[a-zA-Z]+\s+[가-힣]+(?:\s+[가-힣가-힣a-zA-Z0-9]+)*', 0),  # Space-separated
            (r'[a-zA-Z]+[가-힣]+(?:[0-9]+)?', 0),  # Mixed with numbers
            (r'[a-zA-Z]+[가-힣]+\s+[가-힣]+', 0),  # Two words
            (r'[a-zA-Z]+[가-힣]+\s+[가-힣]+\s+[가-힣]+', 0),  # Three words
            (r'[a-zA-Z가-힣]+(?:\s+[a-zA-Z가-힣]+){1,3}', 0),  # Brand + product
            (r'\d+\s+[a-zA-Z]+', 0),  # Number + English
            (r'[a-zA-Z가-힣0-9]+(?:\s+[a-zA-Z가-힣0-9]+)*', 0),  # General mixed
        ]
        
        # Apply patterns
        for pattern, _ in patterns:
            for match in re.finditer(pattern, text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
        
        # Add individual tokens
        for span in re.split(r'[,\.!?;:"\'…\(\)\[\]\{\}\s_/]+', text):
            if span and len(span) >= min_len:
                span_pos = text.find(span)
                if span_pos != -1:
                    spans.append((span, span_pos, span_pos + len(span)))
        
        return spans
    
    def _resolve_overlapping_matches(
        self,
        matches: List[Dict[str, Any]],
        high_score_threshold: int = ENTITY_MATCHING_CONFIG["high_score_threshold"],
        overlap_tolerance: float = ENTITY_MATCHING_CONFIG["overlap_tolerance"]
    ) -> List[Dict[str, Any]]:
        """Resolve overlapping matches by keeping the best ones."""
        if not matches:
            return []
        
        # Sort by score and length
        sorted_matches = sorted(
            matches,
            key=lambda x: (-x['score'], x['end'] - x['start'])
        )
        
        final_matches = []
        
        for current_match in sorted_matches:
            current_score = current_match['score']
            current_start, current_end = current_match['start'], current_match['end']
            current_range = set(range(current_start, current_end))
            current_len = len(current_range)
            
            current_match['overlap_ratio'] = 0.0
            
            # High score matches
            if current_score >= high_score_threshold:
                is_too_similar = False
                
                for existing_match in final_matches:
                    if existing_match['score'] < high_score_threshold:
                        continue
                    
                    existing_range = set(range(
                        existing_match['start'],
                        existing_match['end']
                    ))
                    
                    intersection = current_range.intersection(existing_range)
                    current_overlap_ratio = len(intersection) / current_len if current_len > 0 else 0
                    
                    current_match['overlap_ratio'] = max(
                        current_match['overlap_ratio'],
                        current_overlap_ratio
                    )
                    
                    if (current_overlap_ratio > overlap_tolerance and
                        current_match['matched_entity'] == existing_match['matched_entity']):
                        is_too_similar = True
                        break
                
                if not is_too_similar:
                    final_matches.append(current_match)
            
            # Low score matches
            else:
                should_add = True
                
                for existing_match in final_matches:
                    existing_range = set(range(
                        existing_match['start'],
                        existing_match['end']
                    ))
                    
                    intersection = current_range.intersection(existing_range)
                    current_overlap_ratio = len(intersection) / current_len if current_len > 0 else 0
                    
                    current_match['overlap_ratio'] = max(
                        current_match['overlap_ratio'],
                        current_overlap_ratio
                    )
                    
                    if current_overlap_ratio > (1 - overlap_tolerance):
                        should_add = False
                        break
                
                if should_add:
                    final_matches.append(current_match)
        
        # Sort by position
        final_matches.sort(key=lambda x: x['start'])
        return final_matches


def find_entities_in_text(
    text: str,
    entity_list: List[Tuple[str, Dict[str, Any]]],
    min_similarity: int = ENTITY_MATCHING_CONFIG["min_similarity"],
    ngram_size: int = ENTITY_MATCHING_CONFIG["ngram_size"],
    min_entity_length: int = ENTITY_MATCHING_CONFIG["min_entity_length"],
    token_similarity: bool = ENTITY_MATCHING_CONFIG["token_similarity"],
    high_score_threshold: int = ENTITY_MATCHING_CONFIG["high_score_threshold"],
    overlap_tolerance: float = ENTITY_MATCHING_CONFIG["overlap_tolerance"]
) -> List[Dict[str, Any]]:
    """
    Find entity matches in text using fuzzy matching.
    
    Args:
        text: Text to search for entities
        entity_list: List of (entity_name, data) tuples
        min_similarity: Minimum similarity score (0-100)
        ngram_size: Size of character n-grams
        min_entity_length: Minimum entity length
        token_similarity: Whether to use token-based similarity
        high_score_threshold: Score threshold for high-confidence matches
        overlap_tolerance: Overlap tolerance ratio
        
    Returns:
        List of matched entities with position and metadata
    """
    matcher = KoreanEntityMatcher(
        min_similarity=min_similarity,
        ngram_size=ngram_size,
        min_entity_length=min_entity_length,
        token_similarity=token_similarity
    )
    matcher.build_from_list(entity_list)
    
    matches = matcher.find_entities(text)
    return matcher._resolve_overlapping_matches(
        matches,
        high_score_threshold=high_score_threshold,
        overlap_tolerance=overlap_tolerance
    ) 