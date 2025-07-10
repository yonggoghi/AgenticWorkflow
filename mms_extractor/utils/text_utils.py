"""
Text utility functions for the MMS extractor.
"""

from typing import List, Set, Dict, Any, Optional
import re
import unicodedata
from collections import Counter

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def tokenize_korean(text: str) -> List[str]:
    """
    Tokenize Korean text into words.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Normalize text
    text = normalize_text(text)
    
    # Split into Korean and non-Korean tokens
    tokens = re.findall(
        r'[가-힣]+|[a-zA-Z]+|\d+',
        text
    )
    
    return [token for token in tokens if token]

def extract_ngrams(
    text: str,
    n: int = 2,
    include_original: bool = True
) -> List[str]:
    """
    Extract n-grams from text.
    
    Args:
        text: Input text
        n: N-gram size
        include_original: Whether to include original text
        
    Returns:
        List of n-grams
    """
    if not text:
        return []
    
    # Normalize text
    text = normalize_text(text)
    
    # Get characters
    chars = list(text)
    
    # Generate n-grams
    ngrams = []
    for i in range(len(chars) - n + 1):
        ngram = ''.join(chars[i:i + n])
        ngrams.append(ngram)
    
    # Add original text if requested
    if include_original and len(text) >= n:
        ngrams.append(text)
    
    return ngrams

def calculate_similarity(
    text1: str,
    text2: str,
    method: str = "jaccard",
    ngram_size: int = 2
) -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method (jaccard, cosine, or levenshtein)
        ngram_size: Size of n-grams for jaccard and cosine
        
    Returns:
        Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    if method == "jaccard":
        # Get n-grams
        ngrams1 = set(extract_ngrams(text1, n=ngram_size))
        ngrams2 = set(extract_ngrams(text2, n=ngram_size))
        
        # Calculate Jaccard similarity
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
        
    elif method == "cosine":
        # Get n-grams
        ngrams1 = extract_ngrams(text1, n=ngram_size)
        ngrams2 = extract_ngrams(text2, n=ngram_size)
        
        # Calculate term frequencies
        freq1 = Counter(ngrams1)
        freq2 = Counter(ngrams2)
        
        # Get unique terms
        terms = set(freq1.keys()).union(set(freq2.keys()))
        
        # Calculate dot product and magnitudes
        dot_product = sum(freq1[term] * freq2[term] for term in terms)
        mag1 = sum(freq1[term] ** 2 for term in terms) ** 0.5
        mag2 = sum(freq2[term] ** 2 for term in terms) ** 0.5
        
        # Calculate cosine similarity
        return dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0.0
        
    elif method == "levenshtein":
        # Calculate Levenshtein distance
        m, n = len(text1), len(text2)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if text1[i - 1] == text2[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,  # deletion
                        d[i][j - 1] + 1,  # insertion
                        d[i - 1][j - 1] + 1  # substitution
                    )
        
        # Convert distance to similarity
        max_len = max(m, n)
        return 1 - (d[m][n] / max_len) if max_len > 0 else 0.0
        
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def find_best_match(
    text: str,
    candidates: List[str],
    method: str = "jaccard",
    ngram_size: int = 2,
    min_similarity: float = 0.5
) -> Optional[tuple[str, float]]:
    """
    Find best matching candidate for text.
    
    Args:
        text: Input text
        candidates: List of candidate texts
        method: Similarity method
        ngram_size: Size of n-grams
        min_similarity: Minimum similarity threshold
        
    Returns:
        Tuple of (best match, similarity score) or None
    """
    if not text or not candidates:
        return None
    
    # Calculate similarities
    similarities = [
        (candidate, calculate_similarity(text, candidate, method, ngram_size))
        for candidate in candidates
    ]
    
    # Find best match
    best_match = max(similarities, key=lambda x: x[1])
    
    # Return if above threshold
    return best_match if best_match[1] >= min_similarity else None

def extract_keywords(
    text: str,
    stop_words: Optional[Set[str]] = None,
    min_length: int = 2,
    max_length: int = 20,
    min_freq: int = 1
) -> List[tuple[str, int]]:
    """
    Extract keywords from text.
    
    Args:
        text: Input text
        stop_words: Set of stop words to exclude
        min_length: Minimum keyword length
        max_length: Maximum keyword length
        min_freq: Minimum frequency threshold
        
    Returns:
        List of (keyword, frequency) tuples
    """
    if not text:
        return []
    
    # Tokenize text
    tokens = tokenize_korean(text)
    
    # Filter tokens
    if stop_words:
        tokens = [
            token for token in tokens
            if (min_length <= len(token) <= max_length and
                token.lower() not in stop_words)
        ]
    else:
        tokens = [
            token for token in tokens
            if min_length <= len(token) <= max_length
        ]
    
    # Count frequencies
    freq = Counter(tokens)
    
    # Filter by frequency
    keywords = [
        (token, count)
        for token, count in freq.items()
        if count >= min_freq
    ]
    
    # Sort by frequency
    return sorted(keywords, key=lambda x: (-x[1], x[0]))

def clean_html(
    text: str,
    remove_tags: Optional[Set[str]] = None,
    remove_attrs: bool = True
) -> str:
    """
    Clean HTML from text.
    
    Args:
        text: Input text
        remove_tags: Set of tags to remove
        remove_attrs: Whether to remove attributes
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    if remove_attrs:
        # Remove attributes
        text = re.sub(r'<[^>]+>', lambda m: re.sub(r'\s+[^>]+', '', m.group()), text)
    
    if remove_tags:
        # Remove specified tags
        for tag in remove_tags:
            text = re.sub(
                f'<{tag}[^>]*>.*?</{tag}>',
                ' ',
                text,
                flags=re.IGNORECASE | re.DOTALL
            )
            text = re.sub(
                f'<{tag}[^>]*/>',
                ' ',
                text,
                flags=re.IGNORECASE
            )
    
    # Remove remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def extract_entities(
    text: str,
    patterns: Dict[str, str],
    min_length: int = 2,
    max_length: int = 50
) -> Dict[str, List[str]]:
    """
    Extract entities from text using regex patterns.
    
    Args:
        text: Input text
        patterns: Dictionary of (entity_type, pattern) pairs
        min_length: Minimum entity length
        max_length: Maximum entity length
        
    Returns:
        Dictionary of (entity_type, entities) pairs
    """
    if not text or not patterns:
        return {}
    
    # Normalize text
    text = normalize_text(text)
    
    # Extract entities
    entities = {}
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        entities[entity_type] = [
            match.group().strip()
            for match in matches
            if min_length <= len(match.group().strip()) <= max_length
        ]
    
    return entities 