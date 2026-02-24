"""
ProductMatcherTool — MongoDB offer_master_data에서 유사도 매칭.
"""
import re
import logging
from typing import List
import pandas as pd
from langchain_core.tools import tool

from utils.similarity_utils import (
    parallel_fuzzy_similarity,
    parallel_seq_similarity,
)
from utils.mongodb_utils import MongoDBManager
from config.settings import PROCESSING_CONFIG

logger = logging.getLogger(__name__)

# Module-level cache for item data
_item_data_cache = None


def _build_synonym_map(alias_rule_set: list) -> dict:
    """Build token → canonical_form mapping from single-word partial B-direction rules.

    e.g., "갤럭시 ↔ Galaxy" partial rules → {갤럭시: 'GALAXY', Galaxy: 'GALAXY', ...}
    Used to detect synonym-doublets in generated aliases (e.g., "Galaxy 갤럭시 퀀텀4").
    """
    from itertools import combinations

    # Collect synonym pairs: (a, b) where each is a single word and there are partial rules both ways
    pair_set = set()
    for alias_from, alias_to, alias_type, case_1, case_2 in alias_rule_set:
        if alias_type == 'partial' and ' ' not in alias_from and ' ' not in alias_to:
            # single-word partial rule — treat as synonym pair
            pair_set.add((alias_from, alias_to))

    # Build Union-Find synonym groups
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pair_set:
        union(a, b)

    # Map each word to its canonical (root representative)
    all_words = {w for pair in pair_set for w in pair}
    return {w: find(w) for w in all_words}


def _build_alias_rule_set(alias_rules_raw: list) -> list:
    """alias_rules raw list → (alias_from, alias_to, type, case_1, case_2) 튜플 리스트."""
    if not alias_rules_raw:
        return []

    alias_pdf = pd.DataFrame(alias_rules_raw)
    for col in ('case_1', 'case_2'):
        if col not in alias_pdf.columns:
            alias_pdf[col] = 'S'
    alias_pdf['case_1'] = alias_pdf['case_1'].fillna('S')
    alias_pdf['case_2'] = alias_pdf['case_2'].fillna('S')

    alias_pdf['alias_1'] = alias_pdf['alias_1'].astype(str).str.split("&&")
    alias_pdf['alias_2'] = alias_pdf['alias_2'].astype(str).str.split("&&")
    alias_pdf = alias_pdf.explode('alias_1').explode('alias_2')
    alias_pdf['alias_1'] = alias_pdf['alias_1'].str.strip()
    alias_pdf['alias_2'] = alias_pdf['alias_2'].str.strip()

    # Add reverse rows for bidirectional rules
    bidir_mask = alias_pdf['direction'] == 'B'
    if bidir_mask.any():
        bidir = alias_pdf[bidir_mask].rename(
            columns={'alias_1': 'alias_2', 'alias_2': 'alias_1', 'case_1': 'case_2', 'case_2': 'case_1'}
        )[alias_pdf.columns]
        alias_pdf = pd.concat([alias_pdf, bidir], ignore_index=True)

    return list(zip(
        alias_pdf['alias_1'],
        alias_pdf['alias_2'],
        alias_pdf['type'],
        alias_pdf['case_1'],
        alias_pdf['case_2'],
    ))


def _has_synonym_doublet(text: str, synonym_map: dict) -> bool:
    """Return True if text contains adjacent tokens that normalize to the same canonical form.

    e.g., "Galaxy 갤럭시 퀀텀4" → tokens ["Galaxy","갤럭시","퀀텀4"]
          canonical ["갤럭시_root","갤럭시_root","퀀텀4"] → adjacent duplicate → True
    """
    tokens = text.split()
    if len(tokens) < 2:
        return False
    norm = [synonym_map.get(t, t.lower()) for t in tokens]
    return any(norm[i] == norm[i + 1] for i in range(len(norm) - 1))


def _cascade_aliases_for_item(item_nm: str, alias_rule_set: list, synonym_map: dict, max_depth: int = 7) -> list:
    """BFS cascade: item_nm에서 alias_rules를 연쇄 적용하여 모든 alias 생성.

    Args:
        synonym_map: token → canonical form (from _build_synonym_map).
                     Used to reject generated aliases with synonym-doublet tokens
                     (e.g., "Galaxy 갤럭시 퀀텀4" → both normalize to same → discard).
    """
    if not isinstance(item_nm, str) or not item_nm:
        return []

    processed = set()
    result_set = {item_nm}
    to_process = [(item_nm, 0, frozenset())]

    while to_process:
        current_item, depth, path_applied_rules = to_process.pop(0)

        if depth >= max_depth or current_item in processed:
            continue

        processed.add(current_item)

        for alias_from, alias_to, alias_type, case_1, case_2 in alias_rule_set:
            rule_key = (alias_from, alias_to, alias_type)
            if rule_key in path_applied_rules:
                continue

            if alias_type == 'exact':
                matched = (current_item.lower() == alias_from.lower()) if case_1 == 'I' else (current_item == alias_from)
            else:  # partial
                matched = (alias_from.lower() in current_item.lower()) if case_1 == 'I' else (alias_from in current_item)

            if matched:
                if alias_type == 'exact':
                    new_items = [alias_to.strip()]
                else:
                    if case_1 == 'I':
                        pattern = re.compile(re.escape(alias_from), re.IGNORECASE)
                        new_items = [pattern.sub(alias_to.strip(), current_item, count=1)]
                    else:
                        new_items = [current_item.replace(alias_from.strip(), alias_to.strip())]

                for new_item in new_items:
                    if new_item and new_item not in result_set:
                        # Reject if generated alias has synonym-doublet tokens
                        # e.g., "Galaxy 갤럭시 퀀텀4" (both are "갤럭시" synonyms)
                        if _has_synonym_doublet(new_item, synonym_map):
                            continue
                        result_set.add(new_item)
                        to_process.append((new_item, depth + 1, path_applied_rules | {rule_key}))

    return list(result_set)


def _apply_alias_cascade_and_explode(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. MongoDB의 item_nm_alias explode (기존 alias)
    2. alias_rules cascade로 추가 alias 생성
    3. 두 결과 합산 후 flat DataFrame 반환
    """
    from utils.rules_cache import RulesCache

    # Step 1: Explode existing aliases from MongoDB
    if 'item_nm_alias' in df.columns:
        df_exploded = df.explode('item_nm_alias').reset_index(drop=True)
        df_exploded['item_nm_alias'] = df_exploded['item_nm_alias'].fillna(df_exploded['item_nm'])
    else:
        df_exploded = df.copy()
        df_exploded['item_nm_alias'] = df_exploded['item_nm']

    # Step 2: Build alias rule set
    alias_rules_raw = RulesCache.get_alias_rules()
    alias_rule_set = _build_alias_rule_set(alias_rules_raw)

    if not alias_rule_set:
        logger.warning("[ProductMatcher] alias_rules empty, cascade skipped")
        return df_exploded

    logger.info(f"[ProductMatcher] alias_rules: {len(alias_rule_set)}개 규칙 로드")

    # Build synonym map once for doublet detection across all items
    synonym_map = _build_synonym_map(alias_rule_set)
    logger.info(f"[ProductMatcher] synonym_map: {len(synonym_map)}개 토큰 매핑")

    # Step 3: Compute cascade aliases per unique item_nm
    existing_per_item = df_exploded.groupby('item_nm')['item_nm_alias'].apply(set).to_dict()
    unique_item_nms = df_exploded['item_nm'].dropna().unique().tolist()

    extra_rows = []
    cascade_added_total = 0
    for item_nm in unique_item_nms:
        cascade_aliases = _cascade_aliases_for_item(item_nm, alias_rule_set, synonym_map, max_depth=7)
        existing = existing_per_item.get(item_nm, set())
        new_aliases = [a for a in cascade_aliases if a not in existing]
        if not new_aliases:
            continue

        cascade_added_total += len(new_aliases)
        # Replicate one row per (item_nm, item_id) combination for each new alias
        dedup_cols = [c for c in ['item_nm', 'item_id'] if c in df_exploded.columns]
        base_rows = df_exploded[df_exploded['item_nm'] == item_nm].drop_duplicates(subset=dedup_cols)
        for alias in new_aliases:
            for _, base_row in base_rows.iterrows():
                new_row = base_row.to_dict()
                new_row['item_nm_alias'] = alias
                extra_rows.append(new_row)

    if not extra_rows:
        return df_exploded

    extra_df = pd.DataFrame(extra_rows)
    result = pd.concat([df_exploded, extra_df], ignore_index=True)
    dedup_subset = [c for c in ['item_nm', 'item_nm_alias', 'item_id'] if c in result.columns]
    result = result.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    logger.info(f"[ProductMatcher] cascade alias +{cascade_added_total}건 → total {len(result)} alias rows")
    return result


def _load_item_data() -> pd.DataFrame:
    """Load item data from MongoDB (cached), with alias cascade applied."""
    global _item_data_cache
    if _item_data_cache is not None:
        return _item_data_cache

    mgr = MongoDBManager()
    if mgr.connect():
        items = mgr.get_all_item_aliases()
        mgr.disconnect()
        if items:
            df = pd.DataFrame(items)
            df_exploded = _apply_alias_cascade_and_explode(df)
            _item_data_cache = df_exploded
            logger.info(f"[ProductMatcher] loaded {len(df)} items → {len(df_exploded)} alias rows (with cascade)")
            return _item_data_cache

    logger.warning("[ProductMatcher] MongoDB unavailable, returning empty DataFrame")
    _item_data_cache = pd.DataFrame()
    return _item_data_cache


@tool
def match_products(entities: List[str], message: str) -> dict:
    """후보 개체명을 MongoDB offer_master_data에서 검색하여 상품 ID와 매칭합니다.

    Args:
        entities: 후보 개체명 리스트
        message: 원본 MMS 메시지 (컨텍스트 참조용)

    Returns:
        dict with matched_products (list of match dicts with item_nm, item_id, similarity, entity)
    """
    if not entities:
        return {"matched_products": []}

    item_df = _load_item_data()
    if item_df.empty:
        return {"matched_products": []}

    all_aliases = item_df['item_nm_alias'].dropna().unique().tolist()
    threshold = PROCESSING_CONFIG.entity_fuzzy_threshold

    # Step 1: Fuzzy similarity
    fuzzy_df = parallel_fuzzy_similarity(
        texts=entities,
        entities=all_aliases,
        threshold=threshold,
        text_col_nm='entity',
        item_col_nm='item_nm_alias',
    )

    if fuzzy_df.empty:
        return {"matched_products": []}

    # Step 2: Sequence similarity on fuzzy-filtered pairs
    seq_df = parallel_seq_similarity(
        fuzzy_df,
        text_col_nm='entity',
        item_col_nm='item_nm_alias',
        normalization_value='s2',
    )

    # Step 3: Merge and combine scores
    if not seq_df.empty:
        merged = fuzzy_df.merge(
            seq_df, on=['entity', 'item_nm_alias'], suffixes=('_fuzzy', '_seq')
        )
        merged['combined_sim'] = merged['sim_fuzzy'] + merged['sim_seq']
    else:
        merged = fuzzy_df.copy()
        merged['combined_sim'] = merged['sim']

    # Step 4: Per-entity specificity-aware filtering
    # Strategy: prefer specific alias matches over generic ones, then top-K
    high_threshold = PROCESSING_CONFIG.entity_high_similarity_threshold
    filtered_parts = []
    for ent, grp in merged.groupby('entity'):
        grp_sorted = grp.sort_values('combined_sim', ascending=False)

        # Check if entity has exact alias matches (entity text appears in alias or vice versa)
        ent_lower = ent.lower().strip()
        exact_mask = grp_sorted['item_nm_alias'].str.lower().str.strip() == ent_lower
        specific_mask = grp_sorted['item_nm_alias'].apply(
            lambda a: ent_lower in a.lower() or a.lower().strip() in ent_lower
        )
        # Compute alias specificity: ratio of len(alias)/len(entity), prefer ~1.0
        grp_sorted = grp_sorted.copy()
        grp_sorted['alias_len_ratio'] = grp_sorted['item_nm_alias'].str.len() / max(len(ent), 1)

        if exact_mask.any():
            # Exact alias match exists → keep only items matched via this exact alias
            kept = grp_sorted[exact_mask].head(5)
        elif specific_mask.any():
            # Specific match (substring containment) → prefer more specific aliases
            specific_sorted = grp_sorted[specific_mask].sort_values('combined_sim', ascending=False)
            # Sub-filter by alias specificity: keep aliases whose len ratio >= 70% of the best
            max_ratio = specific_sorted['alias_len_ratio'].max()
            if max_ratio > 0:
                ratio_threshold = max_ratio * 0.70
                specific_sorted = specific_sorted[specific_sorted['alias_len_ratio'] >= ratio_threshold]
            best_score = specific_sorted['combined_sim'].iloc[0]
            kept = specific_sorted[specific_sorted['combined_sim'] >= best_score * 0.90].head(5)
        else:
            # No specific match → relative cutoff, top-5
            best_score = grp_sorted['combined_sim'].iloc[0]
            relative_cutoff = best_score * 0.85
            effective_threshold = max(high_threshold, relative_cutoff)
            kept = grp_sorted[grp_sorted['combined_sim'] >= effective_threshold].head(5)

        if kept.empty:
            kept = grp_sorted.head(3)
        filtered_parts.append(kept)

    filtered = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else merged.head(0)

    # Step 4.5: Build entity cascade map for disambiguation.
    # Multiple items in the DB can share the same alias (e.g., "아이폰 17" is registered
    # as a related-product alias for iPhone 17 Pro and Pro Max).
    # We use entity-side cascade expansion to identify the *direct* target item_nm
    # and prefer it over items that only have the entity as a distant related alias.
    from utils.rules_cache import RulesCache
    _raw_rules = RulesCache.get_alias_rules()
    _rs = _build_alias_rule_set(_raw_rules)
    _sm = _build_synonym_map(_rs)
    entity_cascade_map: dict = {}
    for ent in {row['entity'] for _, row in filtered.iterrows()}:
        expanded = _cascade_aliases_for_item(ent, _rs, _sm, max_depth=5)
        entity_cascade_map[ent] = frozenset(e.lower().strip() for e in expanded)

    # Step 5: Map back to item_nm and item_id
    results = []
    for _, row in filtered.iterrows():
        alias = row['item_nm_alias']
        entity = row['entity']
        sim = row['combined_sim']

        # Find all items that have this alias
        item_rows = item_df[item_df['item_nm_alias'] == alias].drop_duplicates(
            subset=['item_nm', 'item_id']
        )

        # Disambiguate: when multiple items share the same alias, prefer the item
        # whose item_nm is directly reachable from the entity via alias cascade.
        # e.g., entity "아이폰 17" cascades to "iPhone 17" but NOT "iPhone 17 Pro",
        # so we drop iPhone 17 Pro/Max even though they all share alias "아이폰 17".
        if len(item_rows) > 1:
            ent_expanded = entity_cascade_map.get(entity, frozenset({entity.lower().strip()}))
            direct_mask = item_rows['item_nm'].str.lower().str.strip().isin(ent_expanded)
            if direct_mask.any():
                logger.info(
                    f"[ProductMatcher] Disambiguate {entity!r}→{alias!r}: "
                    f"kept {list(item_rows[direct_mask]['item_nm'])} "
                    f"(dropped {list(item_rows[~direct_mask]['item_nm'])})"
                )
                item_rows = item_rows[direct_mask]

        for _, item_row in item_rows.iterrows():
            results.append({
                'entity': entity,
                'item_nm': item_row.get('item_nm', alias),
                'item_id': item_row.get('item_id', ''),
                'item_dmn': item_row.get('item_dmn', ''),
                'item_nm_alias': alias,
                'similarity': round(sim, 3),
            })

    # Deduplicate by (entity, item_nm, item_id)
    seen = set()
    unique_results = []
    for r in results:
        key = (r['entity'], r['item_nm'], r['item_id'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    logger.info(f"[ProductMatcher] {len(entities)} entities → {len(unique_results)} product matches")
    return {"matched_products": unique_results}
