import os
import logging
import re
import difflib
import pandas as pd
from rapidfuzz import fuzz
from joblib import Parallel, delayed
from typing import List, Dict, Any, Union

from .text_utils import preprocess_text, replace_special_chars_with_space

logger = logging.getLogger(__name__)

def _extract_bigrams(s):
    """문자열에서 bigram(연속 2글자) 집합 추출 (공백/특수문자 제거 후)"""
    cleaned = re.sub(r'[^가-힣a-zA-Z0-9]', '', s.lower())
    if len(cleaned) < 2:
        return set(cleaned)  # 1글자면 단일 문자를 반환
    return set(cleaned[i:i+2] for i in range(len(cleaned) - 1))

def _build_bigram_index(entities):
    """엔티티 목록에 대한 bigram 인덱스 구축 (bigram → entity indices)"""
    entity_bigrams = []
    bigram_to_entities = {}
    for idx, entity in enumerate(entities):
        bgs = _extract_bigrams(entity)
        entity_bigrams.append(bgs)
        for bg in bgs:
            if bg not in bigram_to_entities:
                bigram_to_entities[bg] = []
            bigram_to_entities[bg].append(idx)
    return entity_bigrams, bigram_to_entities

def _filter_entities_by_bigram(text, entities, bigram_to_entities):
    """텍스트와 bigram이 겹치는 엔티티만 필터링"""
    text_bigrams = _extract_bigrams(text)
    if not text_bigrams:
        return entities  # 빈 텍스트면 필터링 안 함

    candidate_indices = set()
    for bg in text_bigrams:
        if bg in bigram_to_entities:
            candidate_indices.update(bigram_to_entities[bg])

    if not candidate_indices:
        return []

    return [entities[i] for i in sorted(candidate_indices)]

def calculate_fuzzy_similarity(text, entities):
    """퍼지 매칭을 사용한 유사도 계산"""
    results = []
    for entity in entities:
        scores = {
            'ratio': fuzz.ratio(text, entity) / 100,
            'partial_ratio': fuzz.partial_ratio(text, entity) / 100,
            'token_sort_ratio': fuzz.token_sort_ratio(text, entity) / 100,
            'token_set_ratio': fuzz.token_set_ratio(text, entity) / 100
        }
        max_score = max(scores.values())
        results.append((entity, max_score))
    return results

def calculate_fuzzy_similarity_batch(args_dict):
    """배치 처리를 위한 퍼지 유사도 계산 함수"""
    text = args_dict['text']
    entities = args_dict['entities']
    threshold = args_dict['threshold']
    text_col_nm = args_dict['text_col_nm']
    item_col_nm = args_dict['item_col_nm']

    text_processed = preprocess_text(text)
    similarities = calculate_fuzzy_similarity(text_processed, entities)

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

def parallel_fuzzy_similarity(texts, entities, threshold=0.5, text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=None, batch_size=None):
    """병렬 처리를 통한 퍼지 유사도 계산 (bigram 사전 필터링 적용)"""
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)

    # Build bigram index once for all entities
    entities_list = list(entities)
    _, bigram_to_entities = _build_bigram_index(entities_list)

    # Pre-filter entities per text using bigram overlap → one batch per text
    batches = []
    total_candidates = 0
    for text in texts:
        text_processed = preprocess_text(text)
        filtered = _filter_entities_by_bigram(text_processed, entities_list, bigram_to_entities)
        total_candidates += len(filtered)
        if not filtered:
            continue
        batches.append({"text": text, "entities": filtered, "threshold": threshold, "text_col_nm": text_col_nm, "item_col_nm": item_col_nm})

    logger.info(f"Bigram pre-filter: {len(entities_list)} x {len(texts)} → {total_candidates} candidates ({total_candidates/(len(entities_list)*max(len(texts),1))*100:.1f}%)")

    if not batches:
        return pd.DataFrame()

    if total_candidates < 100000:
        # Small candidate set after filtering: single-process avoids joblib IPC overhead
        batch_results = [calculate_fuzzy_similarity_batch(args) for args in batches]
    else:
        with Parallel(n_jobs=n_jobs) as parallel:
            batch_results = parallel(delayed(calculate_fuzzy_similarity_batch)(args) for args in batches)

    return pd.DataFrame(sum(batch_results, []))

def longest_common_subsequence_ratio(s1, s2, normalization_value):
    """최장 공통 부분수열 비율 계산"""
    def lcs_length(x, y):
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

def sequence_matcher_similarity(s1, s2, normalization_value):
    """SequenceMatcher를 사용한 유사도 계산"""
    matcher = difflib.SequenceMatcher(None, s1, s2)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())

    normalization_length = min(len(s1), len(s2))
    if normalization_value == 'max':
        normalization_length = max(len(s1), len(s2))
    elif normalization_value == 's1':
        normalization_length = len(s1)
    elif normalization_value == 's2':
        normalization_length = len(s2)

    if normalization_length == 0:
        return 0.0

    return matches / normalization_length

def substring_aware_similarity(s1, s2, normalization_value):
    """부분문자열 관계를 고려한 유사도 계산"""
    if s1 in s2 or s2 in s1:
        shorter = min(s1, s2, key=len)
        longer = max(s1, s2, key=len)
        base_score = len(shorter) / len(longer)
        return min(0.95 + base_score * 0.05, 1.0)
    return longest_common_subsequence_ratio(s1, s2, normalization_value)

def token_sequence_similarity(s1, s2, normalization_value, separator_pattern=r'[\s_\-]+'):
    """토큰 시퀀스 기반 유사도 계산"""
    tokens1 = [t for t in re.split(separator_pattern, s1.strip()) if t]
    tokens2 = [t for t in re.split(separator_pattern, s2.strip()) if t]

    if not tokens1 or not tokens2:
        return 0.0

    def token_lcs_length(t1, t2):
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
    normalization_tokens = max(len(tokens1), len(tokens2))
    if normalization_value == 'min':
        normalization_tokens = min(len(tokens1), len(tokens2))
    elif normalization_value == 's1':
        normalization_tokens = len(tokens1)
    elif normalization_value == 's2':
        normalization_tokens = len(tokens2)

    return lcs_tokens / normalization_tokens

def combined_sequence_similarity(s1, s2, weights=None, normalization_value='max'):
    """여러 유사도 메트릭을 결합한 종합 유사도 계산"""

    s1 = replace_special_chars_with_space(s1)
    s2 = replace_special_chars_with_space(s2)

    if weights is None:
        weights = {'substring': 0.1, 'sequence_matcher': 0.7, 'token_sequence': 0.2}

    similarities = {
        'substring': substring_aware_similarity(s1, s2, normalization_value),
        'sequence_matcher': sequence_matcher_similarity(s1, s2, normalization_value),
        'token_sequence': token_sequence_similarity(s1, s2, normalization_value)
    }

    return sum(similarities[key] * weights[key] for key in weights), similarities

def calculate_seq_similarity(args_dict):
    """배치 처리를 위한 시퀀스 유사도 계산"""
    sent_item_batch = args_dict['sent_item_batch']
    text_col_nm = args_dict['text_col_nm']
    item_col_nm = args_dict['item_col_nm']
    normalization_value = args_dict['normalization_value']
    weights = args_dict.get('weights', None)
    results = []
    for sent_item in sent_item_batch:
        sent = sent_item[text_col_nm]
        item = sent_item[item_col_nm]
        try:
            sent_processed = preprocess_text(sent.lower())
            item_processed = preprocess_text(item.lower())
            similarity = combined_sequence_similarity(sent_processed, item_processed, weights=weights, normalization_value=normalization_value)[0]
            results.append({text_col_nm:sent, item_col_nm:item, "sim":similarity})
        except Exception as e:
            logger.error(f"Error processing {item}: {e}")
            results.append({text_col_nm:sent, item_col_nm:item, "sim":0.0})

    return results

def parallel_seq_similarity(sent_item_pdf, text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=None, batch_size=None, weights=None, normalization_value='s2'):
    """병렬 처리를 통한 시퀀스 유사도 계산"""
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)

    total_rows = sent_item_pdf.shape[0]

    if total_rows < 5000:
        # Small dataset: single-process avoids joblib IPC overhead
        batch = sent_item_pdf.to_dict(orient='records')
        args = {"sent_item_batch": batch, 'text_col_nm': text_col_nm, 'item_col_nm': item_col_nm, 'weights': weights, 'normalization_value': normalization_value}
        results = calculate_seq_similarity(args)
        return pd.DataFrame(results)

    if batch_size is None:
        batch_size = max(1, total_rows // (n_jobs * 2))

    batches = []
    for i in range(0, total_rows, batch_size):
        batch = sent_item_pdf.iloc[i:i + batch_size].to_dict(orient='records')
        batches.append({"sent_item_batch": batch, 'text_col_nm': text_col_nm, 'item_col_nm': item_col_nm, 'weights': weights, 'normalization_value': normalization_value})

    with Parallel(n_jobs=n_jobs) as parallel:
        batch_results = parallel(delayed(calculate_seq_similarity)(args) for args in batches)

    return pd.DataFrame(sum(batch_results, []))
