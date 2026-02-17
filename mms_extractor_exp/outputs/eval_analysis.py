#!/usr/bin/env python3
"""Comprehensive evaluation analysis for entity extraction CSV."""

import pandas as pd
from difflib import SequenceMatcher
from collections import Counter
import re

CSV_PATH = "/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/outputs/entity_extraction_eval_20260217_143159.csv"

df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# ======================================================================
# TASK 1: Count total entities for each mode
# ======================================================================
print("=" * 80)
print("TASK 1: TOTAL ENTITY COUNTS PER MODE")
print("=" * 80)

EXTRACTED_COLS = [
    "extracted_entities_ax_dag",
    "extracted_entities_ax_ont",
    "extracted_entities_ax_kg",
    "extracted_entities_opus_none",
    "extracted_entities_opus_dag",
    "extracted_entities_ax_dag_v2",
    "correct_extracted_entities",
]

def parse_entities(cell):
    """Parse entities from a cell, splitting by ' | '."""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [e.strip() for e in str(cell).split(" | ") if e.strip()]

for col in EXTRACTED_COLS:
    all_entities = []
    empty_count = 0
    for val in df[col]:
        ents = parse_entities(val)
        if len(ents) == 0:
            empty_count += 1
        all_entities.extend(ents)
    avg = len(all_entities) / len(df) if len(df) > 0 else 0
    print(f"{col}:")
    print(f"  Total entities: {len(all_entities)}, Avg per sample: {avg:.2f}, Empty samples: {empty_count}")

print()

# ======================================================================
# TASK 2: Per-sample matching evaluation at threshold=0.5
# ======================================================================
print("=" * 80)
print("TASK 2: PER-SAMPLE MATCHING EVALUATION (threshold=0.5)")
print("=" * 80)

def compute_matching(extracted_list, gold_list, threshold=0.5):
    """Greedy optimal matching using SequenceMatcher.

    For each extracted entity, find best matching gold entity.
    Use greedy assignment: sort all pairs by similarity descending,
    assign greedily.
    """
    if not extracted_list and not gold_list:
        return 0, 0, 0  # TP, FP, FN
    if not extracted_list:
        return 0, 0, len(gold_list)
    if not gold_list:
        return 0, len(extracted_list), 0

    # Compute all pairwise similarities
    pairs = []
    for i, ext in enumerate(extracted_list):
        for j, gold in enumerate(gold_list):
            ratio = SequenceMatcher(None, ext, gold).ratio()
            pairs.append((ratio, i, j))

    # Sort by similarity descending (greedy optimal)
    pairs.sort(key=lambda x: -x[0])

    matched_ext = set()
    matched_gold = set()
    tp = 0

    for ratio, i, j in pairs:
        if ratio < threshold:
            break
        if i not in matched_ext and j not in matched_gold:
            tp += 1
            matched_ext.add(i)
            matched_gold.add(j)

    fp = len(extracted_list) - tp
    fn = len(gold_list) - tp
    return tp, fp, fn

EVAL_MODES = [
    "extracted_entities_ax_dag",
    "extracted_entities_ax_ont",
    "extracted_entities_ax_kg",
    "extracted_entities_opus_none",
    "extracted_entities_opus_dag",
    "extracted_entities_ax_dag_v2",
]

mode_results = {}

for mode in EVAL_MODES:
    total_tp, total_fp, total_fn = 0, 0, 0
    per_sample_f1 = []

    for idx in range(len(df)):
        ext = parse_entities(df[mode].iloc[idx])
        gold = parse_entities(df["correct_extracted_entities"].iloc[idx])
        tp, fp, fn = compute_matching(ext, gold, threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-sample F1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_sample_f1.append(f1)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    macro_f1 = sum(per_sample_f1) / len(per_sample_f1) if per_sample_f1 else 0

    mode_results[mode] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "macro_f1": macro_f1, "per_sample_f1": per_sample_f1,
    }

    short_name = mode.replace("extracted_entities_", "")
    total_ext = total_tp + total_fp
    total_gold = total_tp + total_fn
    print(f"{short_name}:")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn} | Extracted={total_ext}, Gold={total_gold}")
    print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} (micro)")
    print(f"  Macro-avg F1={macro_f1:.3f}")
    print()

print()

# ======================================================================
# TASK 3: Per-sample comparison ax_dag vs ax_dag_v2
# ======================================================================
print("=" * 80)
print("TASK 3: PER-SAMPLE COMPARISON ax_dag vs ax_dag_v2")
print("=" * 80)

v1_f1s = mode_results["extracted_entities_ax_dag"]["per_sample_f1"]
v2_f1s = mode_results["extracted_entities_ax_dag_v2"]["per_sample_f1"]

improved = []
same = []
degraded = []

for idx in range(len(df)):
    v1 = v1_f1s[idx]
    v2 = v2_f1s[idx]
    if abs(v2 - v1) < 1e-9:
        same.append(idx)
    elif v2 > v1:
        improved.append(idx)
    else:
        degraded.append(idx)

print(f"Improved (v2 > v1): {len(improved)} samples")
print(f"Same (v2 == v1):    {len(same)} samples")
print(f"Degraded (v2 < v1): {len(degraded)} samples")
print()

if degraded:
    print("DEGRADED SAMPLES:")
    print(f"{'Idx':>4} {'v1_F1':>7} {'v2_F1':>7} {'Delta':>7}  Gold_entities -> v1_extracted -> v2_extracted")
    print("-" * 120)
    for idx in degraded:
        v1 = v1_f1s[idx]
        v2 = v2_f1s[idx]
        delta = v2 - v1
        gold = parse_entities(df["correct_extracted_entities"].iloc[idx])
        v1_ext = parse_entities(df["extracted_entities_ax_dag"].iloc[idx])
        v2_ext = parse_entities(df["extracted_entities_ax_dag_v2"].iloc[idx])
        print(f"{idx:>4} {v1:.3f}   {v2:.3f}   {delta:+.3f}   Gold: {gold}")
        print(f"     {'':>7} {'':>7} {'':>7}   v1: {v1_ext}")
        print(f"     {'':>7} {'':>7} {'':>7}   v2: {v2_ext}")
        print()

if improved:
    print("\nIMPROVED SAMPLES (top 15 by delta):")
    improved_sorted = sorted(improved, key=lambda i: v2_f1s[i] - v1_f1s[i], reverse=True)
    print(f"{'Idx':>4} {'v1_F1':>7} {'v2_F1':>7} {'Delta':>7}")
    print("-" * 30)
    for idx in improved_sorted[:15]:
        v1 = v1_f1s[idx]
        v2 = v2_f1s[idx]
        delta = v2 - v1
        print(f"{idx:>4} {v1:.3f}   {v2:.3f}   {delta:+.3f}")

print()

# ======================================================================
# TASK 4: FP analysis for ax_dag_v2
# ======================================================================
print("=" * 80)
print("TASK 4: FALSE POSITIVE ANALYSIS FOR ax_dag_v2")
print("=" * 80)

fp_entities = []  # list of (sample_idx, entity_str)

for idx in range(len(df)):
    ext = parse_entities(df["extracted_entities_ax_dag_v2"].iloc[idx])
    gold = parse_entities(df["correct_extracted_entities"].iloc[idx])

    if not ext:
        continue

    # Compute pairwise similarities
    pairs = []
    for i, e in enumerate(ext):
        for j, g in enumerate(gold):
            ratio = SequenceMatcher(None, e, g).ratio()
            pairs.append((ratio, i, j))

    pairs.sort(key=lambda x: -x[0])

    matched_ext = set()
    matched_gold = set()

    for ratio, i, j in pairs:
        if ratio < 0.5:
            break
        if i not in matched_ext and j not in matched_gold:
            matched_ext.add(i)
            matched_gold.add(j)

    # Unmatched extracted = FP
    for i, e in enumerate(ext):
        if i not in matched_ext:
            fp_entities.append((idx, e))

print(f"\nTotal FP entities: {len(fp_entities)}")
print()

# Categorize FPs
categories = {
    "a) Benefit/discount amounts (할인, %, 원, 무료)": [],
    "b) Sub-feature descriptions": [],
    "c) Channel/app names": [],
    "d) Store/dealer names": [],
    "e) Action descriptions": [],
    "f) Other": [],
}

benefit_patterns = re.compile(r'할인|%|원[) ]|원$|무료|혜택|쿠폰|캐시백|적립|환급|보장|감면|면제|지원금|요금|페이백|리워드|보상|크레딧|포인트')
channel_patterns = re.compile(r'앱|APP|app|카카오|네이버|T전화|T다이렉트|채널|페이지|홈페이지|링크|URL|바로가기|사이트')
store_patterns = re.compile(r'매장|대리점|지점|센터|마켓|스토어|shop|올리브영|스타벅스|이마트|GS25|CU |세븐일레븐')
action_patterns = re.compile(r'가입|신청|변경|해지|구매|예약|응모|참여|확인|이용|설정|다운|접속|클릭')
sub_feature_patterns = re.compile(r'기능|서비스 제공|안내|알림|문자|통화|데이터|로밍|부재중|차단|필터|보안|백업|저장|용량')

for sample_idx, entity in fp_entities:
    if benefit_patterns.search(entity):
        categories["a) Benefit/discount amounts (할인, %, 원, 무료)"].append((sample_idx, entity))
    elif sub_feature_patterns.search(entity):
        categories["b) Sub-feature descriptions"].append((sample_idx, entity))
    elif channel_patterns.search(entity):
        categories["c) Channel/app names"].append((sample_idx, entity))
    elif store_patterns.search(entity):
        categories["d) Store/dealer names"].append((sample_idx, entity))
    elif action_patterns.search(entity):
        categories["e) Action descriptions"].append((sample_idx, entity))
    else:
        categories["f) Other"].append((sample_idx, entity))

for cat_name, items in categories.items():
    print(f"\n{cat_name}: {len(items)} FPs")
    for sample_idx, entity in items[:10]:
        print(f"  [sample {sample_idx}] \"{entity}\"")
    if len(items) > 10:
        print(f"  ... and {len(items) - 10} more")

print()

# ======================================================================
# TASK 5: Comma parsing check
# ======================================================================
print("=" * 80)
print("TASK 5: COMMA PARSING / NUMBER FRAGMENT CHECK")
print("=" * 80)

# Check for fragments that look like broken number parsing
# e.g., "10" alone, "000원", numbers without context,
# or entities that are suspiciously short numeric fragments
fragment_pattern = re.compile(r'^[\d,]+$|^\d{1,3}$|^,?\d{3}원|^\d+원$|^0{2,}')

# Also check: entities that are just numbers, or number fragments from comma-split
# Korean thousand separator uses comma: 10,000원 -> if split by comma -> "10" and "000원"
comma_fragment_pattern = re.compile(r'^(\d{1,3})$|^(\d{3}원.*)$')

problematic_samples = {}

for col in ["extracted_entities_ax_dag_v2", "extracted_entities_ax_dag",
            "extracted_entities_ax_kg", "extracted_entities_ax_ont",
            "extracted_entities_opus_none", "extracted_entities_opus_dag"]:
    issues = []
    for idx in range(len(df)):
        cell = df[col].iloc[idx]
        if pd.isna(cell):
            continue
        entities = parse_entities(cell)
        for e in entities:
            # Check for short numeric-only fragments
            e_stripped = e.strip()
            # Fragment: purely numeric 1-3 digits
            if re.match(r'^\d{1,3}$', e_stripped):
                issues.append((idx, e_stripped))
            # Fragment: starts with 000 (broken thousand separator)
            elif re.match(r'^0{2,}\d*원?', e_stripped):
                issues.append((idx, e_stripped))
            # Fragment: "000원" pattern from comma split
            elif re.match(r'^\d{3}원', e_stripped) and len(e_stripped) <= 5:
                issues.append((idx, e_stripped))

    if issues:
        short = col.replace("extracted_entities_", "")
        print(f"\n{short}: {len(issues)} potential comma-fragment issues")
        for idx, e in issues:
            print(f"  [sample {idx}] \"{e}\"")

# Also check gold standard for reference
print("\n--- Checking gold standard for comma fragments ---")
for idx in range(len(df)):
    gold = parse_entities(df["correct_extracted_entities"].iloc[idx])
    for e in gold:
        e_stripped = e.strip()
        if re.match(r'^\d{1,3}$', e_stripped) or re.match(r'^0{2,}', e_stripped):
            print(f"  [gold, sample {idx}] \"{e_stripped}\"")

# Broader check: any entity that looks like it might have been incorrectly comma-split
# Look for suspiciously short entities (< 3 chars) that are numeric
print("\n--- Broader check: all entities < 4 chars across ax_dag_v2 ---")
for idx in range(len(df)):
    entities = parse_entities(df["extracted_entities_ax_dag_v2"].iloc[idx])
    for e in entities:
        if len(e.strip()) < 4:
            print(f"  [sample {idx}] \"{e.strip()}\"")

# Check if any CSV cells contain commas inside entity names that might cause misparsing
print("\n--- Checking for entities with embedded commas (potential CSV issues) ---")
for col in EXTRACTED_COLS:
    count = 0
    for idx in range(len(df)):
        cell = df[col].iloc[idx]
        if pd.isna(cell):
            continue
        entities = parse_entities(cell)
        for e in entities:
            if ',' in e:
                if count < 5:
                    short = col.replace("extracted_entities_", "")
                    print(f"  [{short}, sample {idx}] \"{e}\"")
                count += 1
    if count > 0:
        short = col.replace("extracted_entities_", "")
        print(f"  Total entities with commas in {short}: {count}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
