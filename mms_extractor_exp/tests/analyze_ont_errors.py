#!/usr/bin/env python3
"""
Analyze ONT evaluation results vs correct answers to categorize error types.
"""

import pandas as pd
from typing import List, Set, Dict, Tuple

# Read CSV files
ont_results = pd.read_csv(
    '/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/outputs/entity_extraction_eval_ont_20260206_104832.csv'
)
correct_answers = pd.read_csv(
    '/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/outputs/entity_extraction_eval_reg_baseline.csv'
)

def parse_entities(entity_str: str) -> Set[str]:
    """Parse entity string into a set of individual entities."""
    if pd.isna(entity_str) or entity_str == '':
        return set()
    # Split by ' | ' and strip whitespace
    return set(e.strip() for e in str(entity_str).split(' | ') if e.strip())

def categorize_false_positive(entity: str, message: str) -> str:
    """Categorize a false positive entity."""
    entity_lower = entity.lower()

    # Category 1: Benefit (금전적 혜택, 할인, 증정 등)
    benefit_keywords = [
        '할인', '증정', '무료', '캐시백', '보상', '쿠폰', '사은품',
        '기프트', '경품', '혜택', '특가', '이벤트', '최대', '선물'
    ]
    if any(kw in entity_lower for kw in benefit_keywords):
        # Check if it's a specific product/service name
        if not any(x in entity_lower for x in ['갤럭시', '아이폰', 'iphone', '워치', '에어팟']):
            return "Benefit"

    # Category 2: Channel (앱, 고객센터, URL, 매장 홈페이지 등)
    channel_keywords = [
        '고객센터', '매장', '홈페이지', 'http', 'https', 'bit.ly',
        'url', '상담', '앱', '카카오톡', '링크', '가입하기',
        '바로 가기', '자세히 보기', '방문'
    ]
    if any(kw in entity_lower for kw in channel_keywords):
        return "Channel"

    # Category 3: Pricing/Amount (금액, 가격 정보)
    pricing_keywords = ['원', '만원', '천원', '가격', '요금', '이용요금']
    # Check if entity contains numbers followed by 원
    if any(kw in entity for kw in pricing_keywords) or (any(c.isdigit() for c in entity) and '원' in entity):
        return "Pricing/Amount"

    # Category 4: Segment/Condition (고객 세그먼트, 약정 조건)
    segment_keywords = [
        '만 ', '세', '고객', '회원', '가입 시', '이용 시', '구매 시',
        '약정', '조건', '선착순', '신규', '기간', '일정'
    ]
    if any(kw in entity for kw in segment_keywords):
        return "Segment/Condition"

    # Category 5: Generic/Vague (너무 일반적인 단어)
    generic_keywords = [
        '5g', '기기', '스마트폰', '신제품', '출시', '개통',
        '구매', '할인', '제공', '서비스', '이용'
    ]
    # Only if it's very short and generic
    if len(entity) < 15 and any(entity_lower == kw or entity_lower.startswith(kw + ' ') for kw in generic_keywords):
        return "Generic/Vague"

    # Category 7: Other (default)
    return "Other"

def categorize_false_negative(entity: str, message: str, ont_entities: Set[str]) -> str:
    """Categorize a false negative (missed entity)."""
    entity_lower = entity.lower()

    # Check if partially captured (variant form exists in ONT results)
    for ont_entity in ont_entities:
        ont_lower = ont_entity.lower()
        # Check for partial matches
        if (entity_lower in ont_lower or ont_lower in entity_lower) and entity_lower != ont_lower:
            return "Entity partially captured (variant form)"

    # Check exact name
    if entity in ont_entities:
        return "Exact entity name not captured"  # This shouldn't happen, but just in case

    return "Completely missed"

# Initialize counters
fp_categories_ax = {
    "Benefit": [],
    "Channel": [],
    "Pricing/Amount": [],
    "Segment/Condition": [],
    "Generic/Vague": [],
    "Correct but format mismatch": [],
    "Other": []
}

fp_categories_cld = {
    "Benefit": [],
    "Channel": [],
    "Pricing/Amount": [],
    "Segment/Condition": [],
    "Generic/Vague": [],
    "Correct but format mismatch": [],
    "Other": []
}

fn_categories_ax = {
    "Exact entity name not captured": [],
    "Entity partially captured (variant form)": [],
    "Completely missed": []
}

fn_categories_cld = {
    "Exact entity name not captured": [],
    "Entity partially captured (variant form)": [],
    "Completely missed": []
}

# Analyze each row
for idx in range(len(ont_results)):
    message = ont_results.iloc[idx]['mms']

    # Parse entities
    ax_entities = parse_entities(ont_results.iloc[idx]['extracted_entities_ax'])
    cld_entities = parse_entities(ont_results.iloc[idx]['extracted_entities_cld'])
    correct_entities = parse_entities(correct_answers.iloc[idx]['correct_extracted_entities'])

    print(f"\n=== Row {idx + 1} ===")
    print(f"AX entities: {len(ax_entities)}")
    print(f"CLD entities: {len(cld_entities)}")
    print(f"Correct entities: {len(correct_entities)}")

    # False Positives for AX
    fp_ax = ax_entities - correct_entities
    for entity in fp_ax:
        category = categorize_false_positive(entity, message)
        fp_categories_ax[category].append((idx + 1, entity))
        print(f"  AX FP [{category}]: {entity}")

    # False Positives for CLD
    fp_cld = cld_entities - correct_entities
    for entity in fp_cld:
        category = categorize_false_positive(entity, message)
        fp_categories_cld[category].append((idx + 1, entity))
        print(f"  CLD FP [{category}]: {entity}")

    # False Negatives for AX
    fn_ax = correct_entities - ax_entities
    for entity in fn_ax:
        category = categorize_false_negative(entity, message, ax_entities)
        fn_categories_ax[category].append((idx + 1, entity))
        print(f"  AX FN [{category}]: {entity}")

    # False Negatives for CLD
    fn_cld = correct_entities - cld_entities
    for entity in fn_cld:
        category = categorize_false_negative(entity, message, cld_entities)
        fn_categories_cld[category].append((idx + 1, entity))
        print(f"  CLD FN [{category}]: {entity}")

# Print summary
print("\n" + "=" * 80)
print("FALSE POSITIVE SUMMARY")
print("=" * 80)

print("\n### AX False Positives ###")
for category, items in fp_categories_ax.items():
    print(f"{category}: {len(items)}")
    for row_num, entity in items[:3]:  # Show first 3 examples
        print(f"  - Row {row_num}: {entity}")
    if len(items) > 3:
        print(f"  ... and {len(items) - 3} more")

print("\n### CLD False Positives ###")
for category, items in fp_categories_cld.items():
    print(f"{category}: {len(items)}")
    for row_num, entity in items[:3]:  # Show first 3 examples
        print(f"  - Row {row_num}: {entity}")
    if len(items) > 3:
        print(f"  ... and {len(items) - 3} more")

print("\n" + "=" * 80)
print("FALSE NEGATIVE SUMMARY")
print("=" * 80)

print("\n### AX False Negatives ###")
for category, items in fn_categories_ax.items():
    print(f"{category}: {len(items)}")
    for row_num, entity in items[:3]:  # Show first 3 examples
        print(f"  - Row {row_num}: {entity}")
    if len(items) > 3:
        print(f"  ... and {len(items) - 3} more")

print("\n### CLD False Negatives ###")
for category, items in fn_categories_cld.items():
    print(f"{category}: {len(items)}")
    for row_num, entity in items[:3]:  # Show first 3 examples
        print(f"  - Row {row_num}: {entity}")
    if len(items) > 3:
        print(f"  ... and {len(items) - 3} more")

# Create summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

print("\n### False Positive Categories ###")
print(f"{'Category':<30} {'AX Count':<10} {'CLD Count':<10}")
print("-" * 50)
for category in fp_categories_ax.keys():
    ax_count = len(fp_categories_ax[category])
    cld_count = len(fp_categories_cld[category])
    print(f"{category:<30} {ax_count:<10} {cld_count:<10}")

print(f"\n{'TOTAL FALSE POSITIVES':<30} {sum(len(v) for v in fp_categories_ax.values()):<10} {sum(len(v) for v in fp_categories_cld.values()):<10}")

print("\n### False Negative Categories ###")
print(f"{'Category':<45} {'AX Count':<10} {'CLD Count':<10}")
print("-" * 65)
for category in fn_categories_ax.keys():
    ax_count = len(fn_categories_ax[category])
    cld_count = len(fn_categories_cld[category])
    print(f"{category:<45} {ax_count:<10} {cld_count:<10}")

print(f"\n{'TOTAL FALSE NEGATIVES':<45} {sum(len(v) for v in fn_categories_ax.values()):<10} {sum(len(v) for v in fn_categories_cld.values()):<10}")
