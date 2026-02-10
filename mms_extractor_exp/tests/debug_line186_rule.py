"""
Debug line 186 rule differences between the two CSVs.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd


def trace_cascade(item_nm: str, alias_rule_set: list, max_depth: int = 3):
    """Trace cascade with focus on line 186 related rules."""

    processed = set()
    result_dict = {item_nm: '#' * len(item_nm)}
    to_process = [(item_nm, 0, frozenset())]

    # Track line 186 related matches
    line186_matches = []

    while to_process:
        current_item, depth, path_applied_rules = to_process.pop(0)

        if depth >= max_depth or current_item in processed:
            continue

        processed.add(current_item)

        for r in alias_rule_set:
            alias_from, alias_to, alias_type = r[0], r[1], r[2]
            rule_key = (alias_from, alias_to, alias_type)

            if rule_key in path_applied_rules:
                continue

            if alias_type == 'exact':
                matched = (current_item == alias_from)
            else:
                matched = (alias_from in current_item)

            if matched:
                new_item = alias_to.strip() if alias_type == 'exact' else current_item.replace(alias_from.strip(), alias_to.strip())

                # Check if this is line 186 related (아이폰 17 exact rules)
                if alias_type == 'exact' and alias_from in ['아이폰 17', '아이폰 17 PRO', '아이폰 17 PRO MAX']:
                    line186_matches.append({
                        'depth': depth,
                        'current_item': current_item,
                        'alias_from': alias_from,
                        'alias_to': alias_to,
                        'new_item': new_item,
                        'already_exists': new_item in result_dict
                    })

                if new_item not in result_dict:
                    result_dict[new_item] = alias_from.strip()
                    to_process.append((new_item, depth + 1, path_applied_rules | {rule_key}))

    return result_dict, line186_matches


def debug_line186():
    """Compare line 186 rule application between two CSVs."""

    # Load alias rules
    alias_pdf = pd.read_csv('./data/alias_rules.csv')
    alias_pdf['alias_1'] = alias_pdf['alias_1'].str.split("&&")
    alias_pdf['alias_2'] = alias_pdf['alias_2'].str.split("&&")
    alias_pdf = alias_pdf.explode('alias_1')
    alias_pdf = alias_pdf.explode('alias_2')

    # Bidirectional
    bidirectional = alias_pdf.query("direction=='B'").rename(
        columns={'alias_1': 'alias_2', 'alias_2': 'alias_1'}
    )[alias_pdf.columns]
    alias_pdf = pd.concat([alias_pdf, bidirectional])

    alias_rule_set = list(zip(
        alias_pdf['alias_1'],
        alias_pdf['alias_2'],
        alias_pdf['type']
    ))

    # Show line 186 related rules
    print("=" * 70)
    print("Line 186 related exact rules (alias_1 contains '아이폰 17'):")
    print("=" * 70)
    line186_rules = alias_pdf[
        (alias_pdf['type'] == 'exact') &
        (alias_pdf['alias_1'].str.contains('아이폰 17', na=False))
    ]
    for _, row in line186_rules.iterrows():
        print(f"  '{row['alias_1']}' → '{row['alias_2']}'")

    # Case 1: Original CSV - 'iPhone 17'
    print("\n" + "=" * 70)
    print("Case 1: Original CSV item_nm = 'iPhone 17'")
    print("=" * 70)

    result1, matches1 = trace_cascade('iPhone 17', alias_rule_set)

    print(f"\nLine 186 rule matches ({len(matches1)}):")
    for m in matches1:
        print(f"  Depth {m['depth']}: '{m['current_item']}' == '{m['alias_from']}' → '{m['new_item']}' (exists: {m['already_exists']})")

    print(f"\nFinal items containing '아이폰17' or '아이폰 17 시리즈':")
    for k in result_dict_items(result1, ['아이폰17', '아이폰 17 시리즈']):
        print(f"  - '{k}'")

    # Case 2: Rev CSV - 'IPHONE 17 PRO MAX'
    print("\n" + "=" * 70)
    print("Case 2: Rev CSV item_nm = 'IPHONE 17 PRO MAX'")
    print("=" * 70)

    result2, matches2 = trace_cascade('IPHONE 17 PRO MAX', alias_rule_set)

    print(f"\nLine 186 rule matches ({len(matches2)}):")
    for m in matches2:
        print(f"  Depth {m['depth']}: '{m['current_item']}' == '{m['alias_from']}' → '{m['new_item']}' (exists: {m['already_exists']})")

    print(f"\nFinal items containing '아이폰17' or '아이폰 17 시리즈':")
    for k in result_dict_items(result2, ['아이폰17', '아이폰 17 시리즈']):
        print(f"  - '{k}'")

    # Key difference
    print("\n" + "=" * 70)
    print("KEY DIFFERENCE:")
    print("=" * 70)

    print(f"\nCase 1 ('iPhone 17'):")
    print(f"  - Can reach '아이폰 17' via partial rule 'iPhone' → '아이폰'")
    print(f"  - '아이폰 17' EXACTLY matches line 186 alias_1")
    print(f"  - Triggers: '아이폰 17' → '아이폰17', '아이폰 17 시리즈'")

    print(f"\nCase 2 ('IPHONE 17 PRO MAX'):")
    print(f"  - Can reach '아이폰 17 PRO MAX' via partial rule 'IPHONE' → '아이폰'")
    has_exact_match = '아이폰 17 PRO MAX' in [r[0] for r in alias_rule_set if r[2] == 'exact']
    print(f"  - '아이폰 17 PRO MAX' exact rule exists: {has_exact_match}")

    # Check what '아이폰 17 PRO MAX' maps to
    promax_rules = [(r[0], r[1]) for r in alias_rule_set if r[0] == '아이폰 17 PRO MAX' and r[2] == 'exact']
    if promax_rules:
        print(f"  - '아이폰 17 PRO MAX' exact rules:")
        for fr, to in promax_rules:
            print(f"      → '{to}'")


def result_dict_items(result_dict, patterns):
    """Find items in result_dict matching any pattern."""
    items = []
    for k in result_dict.keys():
        for p in patterns:
            if p in k or k == p:
                items.append(k)
                break
    return items


if __name__ == '__main__':
    debug_line186()
