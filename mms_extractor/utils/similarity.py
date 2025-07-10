"""
Similarity calculation utilities for text matching.
"""
import re
import difflib
from typing import List, Tuple, Dict, Any
from difflib import SequenceMatcher
import pandas as pd
from rapidfuzz import fuzz, process
from joblib import Parallel, delayed
import os
import numpy as np
from .text_processing import preprocess_text


def longest_common_subsequence_ratio(s1: str, s2: str, normalization_value: str) -> float:
    """
    Calculate similarity based on longest common subsequence (LCS).
    
    Args:
        s1: First string
        s2: Second string
        normalization_value: Normalization method ('max', 'min', 's1', 's2')
        
    Returns:
        LCS ratio
    """
    def lcs_length(x: str, y: str) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_len = lcs_length(s1, s2)
    
    if normalization_value == 'max':
        max_len = max(len(s1), len(s2))
        return lcs_len / max_len if max_len > 0 else 1.0
    elif normalization_value == 'min':
        min_len = min(len(s1), len(s2))
        return lcs_len / min_len if min_len > 0 else 1.0
    elif normalization_value == 's1':
        return lcs_len / len(s1) if len(s1) > 0 else 1.0
    elif normalization_value == 's2':
        return lcs_len / len(s2) if len(s2) > 0 else 1.0
    else:
        raise ValueError(f"Invalid normalization value: {normalization_value}")


def sequence_matcher_similarity(s1: str, s2: str, normalization_value: str) -> float:
    """
    Calculate similarity using SequenceMatcher.
    
    Args:
        s1: First string
        s2: Second string
        normalization_value: Normalization method
        
    Returns:
        Similarity score
    """
    matcher = difflib.SequenceMatcher(None, s1, s2)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())

    if normalization_value == 'max':
        normalization_length = max(len(s1), len(s2))
    elif normalization_value == 'min':
        normalization_length = min(len(s1), len(s2))
    elif normalization_value == 's1':
        normalization_length = len(s1)
    elif normalization_value == 's2':
        normalization_length = len(s2)
    else:
        raise ValueError(f"Invalid normalization value: {normalization_value}")
        
    if normalization_length == 0:
        return 0.0
    
    return matches / normalization_length


def substring_aware_similarity(s1: str, s2: str, normalization_value: str) -> float:
    """
    Custom similarity that heavily weights substring relationships.
    
    Args:
        s1: First string
        s2: Second string
        normalization_value: Normalization method
        
    Returns:
        Similarity score
    """
    # Check if one is a substring of the other
    if s1 in s2 or s2 in s1:
        shorter = min(s1, s2, key=len)
        longer = max(s1, s2, key=len)
        base_score = len(shorter) / len(longer)
        return min(0.95 + base_score * 0.05, 1.0)
    
    return longest_common_subsequence_ratio(s1, s2, normalization_value)


def token_sequence_similarity(s1: str, s2: str, normalization_value: str, 
                             separator_pattern: str = r'[\s_\-]+') -> float:
    """
    Calculate similarity based on token sequence overlap.
    
    Args:
        s1: First string
        s2: Second string
        normalization_value: Normalization method
        separator_pattern: Pattern to split tokens
        
    Returns:
        Token similarity score
    """
    tokens1 = re.split(separator_pattern, s1.strip())
    tokens2 = re.split(separator_pattern, s2.strip())
    
    # Remove empty tokens
    tokens1 = [t for t in tokens1 if t]
    tokens2 = [t for t in tokens2 if t]
    
    if not tokens1 or not tokens2:
        return 0.0
    
    def token_lcs_length(t1: List[str], t2: List[str]) -> int:
        m, n = len(t1), len(t2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if t1[i-1] == t2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_tokens = token_lcs_length(tokens1, tokens2)
    
    if normalization_value == 'max':
        normalization_tokens = max(len(tokens1), len(tokens2))
    elif normalization_value == 'min':
        normalization_tokens = min(len(tokens1), len(tokens2))
    elif normalization_value == 's1':
        normalization_tokens = len(tokens1)
    elif normalization_value == 's2':
        normalization_tokens = len(tokens2)
    else:
        raise ValueError(f"Invalid normalization value: {normalization_value}")
        
    return lcs_tokens / normalization_tokens


def combined_sequence_similarity(s1: str, s2: str, weights: Dict[str, float] = None, 
                               normalization_value: str = 'max') -> Tuple[float, Dict[str, float]]:
    """
    Combine multiple sequence-aware similarity measures.
    
    Args:
        s1: First string
        s2: Second string
        weights: Weights for different similarity measures
        normalization_value: Normalization method
        
    Returns:
        Combined similarity score and individual scores
    """
    if weights is None:
        weights = {
            'substring': 0.4,
            'sequence_matcher': 0.4,
            'token_sequence': 0.2
        }
    
    similarities = {
        'substring': substring_aware_similarity(s1, s2, normalization_value),
        'sequence_matcher': sequence_matcher_similarity(s1, s2, normalization_value),
        'token_sequence': token_sequence_similarity(s1, s2, normalization_value)
    }
    
    combined = sum(similarities[key] * weights[key] for key in weights)
    return combined, similarities


def fuzzy_similarities(text: str, entities: List[str]) -> List[Tuple[str, float]]:
    """
    Calculate fuzzy similarities between text and entities.
    
    Args:
        text: Input text
        entities: List of entities to compare against
        
    Returns:
        List of (entity, score) tuples
    """
    results = []
    for entity in entities:
        try:
            scores = {
                'ratio': fuzz.ratio(text, entity) / 100,
                'partial_ratio': fuzz.partial_ratio(text, entity) / 100,
                'token_sort_ratio': fuzz.token_sort_ratio(text, entity) / 100,
                'token_set_ratio': fuzz.token_set_ratio(text, entity) / 100
            }
            max_score = max(scores.values())
            results.append((entity, max_score))
        except Exception as e:
            # Skip problematic entities
            continue
    return results


def get_fuzzy_similarities(args_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get fuzzy similarities for a batch of comparisons.
    
    Args:
        args_dict: Dictionary containing text, entities, threshold, and column names
        
    Returns:
        List of similarity results
    """
    text = args_dict['text']
    entities = args_dict['entities']
    threshold = args_dict['threshold']
    text_col_nm = args_dict['text_col_nm']
    item_col_nm = args_dict['item_col_nm']

    text_processed = preprocess_text(text.lower())
    similarities = fuzzy_similarities(text_processed, entities)
    
    filtered_results = [
        {
            text_col_nm: text,
            item_col_nm: entity, 
            "sim": score
        } 
        for entity, score in similarities 
        if score >= threshold
    ]
    
    return filtered_results


def parallel_fuzzy_similarity(texts: List[str], entities: List[str], threshold: float = 0.5, 
                            text_col_nm: str = 'sent', item_col_nm: str = 'item_nm_alias', 
                            n_jobs: int = None, batch_size: int = None) -> pd.DataFrame:
    """
    Calculate fuzzy similarities in parallel.
    
    Args:
        texts: List of texts to compare
        entities: List of entities to compare against
        threshold: Minimum similarity threshold
        text_col_nm: Column name for text
        item_col_nm: Column name for items
        n_jobs: Number of parallel jobs
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with similarity results
    """
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)
    
    if batch_size is None:
        batch_size = max(1, len(entities) // (n_jobs * 2))
        
    batches = []
    for text in texts:
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batches.append({
                "text": text, 
                "entities": batch, 
                "threshold": threshold, 
                "text_col_nm": text_col_nm, 
                "item_col_nm": item_col_nm
            })
    
    try:
        with Parallel(n_jobs=n_jobs) as parallel:
            batch_results = parallel(delayed(get_fuzzy_similarities)(args) for args in batches)
        
        return pd.DataFrame(sum(batch_results, []))
    except Exception as e:
        # Fallback to sequential processing
        all_results = []
        for batch in batches:
            all_results.extend(get_fuzzy_similarities(batch))
        return pd.DataFrame(all_results)


def calculate_seq_similarity(args_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate sequence similarity for a batch of items.
    
    Args:
        args_dict: Dictionary containing batch data and parameters
        
    Returns:
        List of similarity results
    """
    sent_item_batch = args_dict['sent_item_batch']
    text_col_nm = args_dict['text_col_nm']
    item_col_nm = args_dict['item_col_nm']
    normalization_value = args_dict['normalization_value']
    
    results = []
    for sent_item in sent_item_batch:
        sent = str(sent_item[text_col_nm])
        item = str(sent_item[item_col_nm])
        try:
            sent_processed = preprocess_text(sent.lower())
            item_processed = preprocess_text(item.lower())
            similarity = combined_sequence_similarity(
                sent_processed, item_processed, 
                normalization_value=normalization_value
            )[0]
            results.append({
                text_col_nm: sent, 
                item_col_nm: item, 
                "sim": float(similarity)
            })
        except Exception as e:
            # Skip problematic items
            results.append({
                text_col_nm: sent, 
                item_col_nm: item, 
                "sim": 0.0
            })
    
    return results


def parallel_seq_similarity(sent_item_pdf: pd.DataFrame, text_col_nm: str = 'sent', 
                          item_col_nm: str = 'item_nm_alias', n_jobs: int = None, 
                          batch_size: int = None, normalization_value: str = 's2') -> pd.DataFrame:
    """
    Calculate sequence similarities in parallel.
    
    Args:
        sent_item_pdf: DataFrame with sentence-item pairs
        text_col_nm: Column name for text
        item_col_nm: Column name for items
        n_jobs: Number of parallel jobs
        batch_size: Batch size for processing
        normalization_value: Normalization method
        
    Returns:
        DataFrame with similarity results
    """
    if sent_item_pdf.empty:
        return pd.DataFrame(columns=[text_col_nm, item_col_nm, 'sim'])
    
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)
    
    if batch_size is None:
        batch_size = max(1, sent_item_pdf.shape[0] // (n_jobs * 2))
        
    batches = []
    for i in range(0, sent_item_pdf.shape[0], batch_size):
        batch = sent_item_pdf.iloc[i:i + batch_size].to_dict(orient='records')
        batches.append({
            "sent_item_batch": batch, 
            'text_col_nm': text_col_nm, 
            'item_col_nm': item_col_nm, 
            'normalization_value': normalization_value
        })
    
    try:
        with Parallel(n_jobs=n_jobs) as parallel:
            batch_results = parallel(delayed(calculate_seq_similarity)(args) for args in batches)
        
        return pd.DataFrame(sum(batch_results, []))
    except Exception as e:
        # Fallback to sequential processing
        all_results = []
        for batch in batches:
            all_results.extend(calculate_seq_similarity(batch))
        return pd.DataFrame(all_results) 