"""
Debug why '아이폰17' (no space) is missing from the alias list.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd
from utils import select_most_comprehensive


def debug_missing_alias():
    """Debug why 아이폰17 is missing."""

    # Load and prepare alias rules
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

    # Check if '아이폰 17' → '아이폰17' rule exists
    print("=" * 60)
    print("1. Check if rule '아이폰 17' → '아이폰17' exists")
    print("=" * 60)

    target_rule = alias_pdf[
        (alias_pdf['alias_1'] == '아이폰 17') &
        (alias_pdf['alias_2'] == '아이폰17')
    ]
    print(f"Found {len(target_rule)} matching rules:")
    print(target_rule[['alias_1', 'alias_2', 'type', 'direction']])

    # Run cascade manually with debug
    print("\n" + "=" * 60)
    print("2. Manual cascade trace")
    print("=" * 60)

    alias_rule_set = list(zip(
        alias_pdf['alias_1'],
        alias_pdf['alias_2'],
        alias_pdf['type']
    ))

    item_nm = 'iPhone 17'
    max_depth = 3
    processed = set()
    result_dict = {item_nm: '#' * len(item_nm)}
    to_process = [(item_nm, 0, frozenset())]

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

                # Debug: check for 아이폰17
                if new_item == '아이폰17' or alias_to == '아이폰17':
                    print(f"\n*** Found '아이폰17' creation ***")
                    print(f"  current_item: '{current_item}'")
                    print(f"  alias_from: '{alias_from}'")
                    print(f"  alias_to: '{alias_to}'")
                    print(f"  alias_type: '{alias_type}'")
                    print(f"  new_item: '{new_item}'")
                    print(f"  already in result_dict: {new_item in result_dict}")

                if new_item not in result_dict:
                    result_dict[new_item] = alias_from.strip()
                    to_process.append((new_item, depth + 1, path_applied_rules | {rule_key}))

    print("\n" + "=" * 60)
    print("3. result_dict contents")
    print("=" * 60)
    for k, v in sorted(result_dict.items()):
        marker = " <-- TARGET" if k == '아이폰17' else ""
        print(f"  '{k}': '{v}'{marker}")

    # Check select_most_comprehensive
    print("\n" + "=" * 60)
    print("4. select_most_comprehensive analysis")
    print("=" * 60)

    item_nm_alias_values = list(result_dict.values())
    print(f"item_nm_alias values: {set(item_nm_alias_values)}")

    selected = select_most_comprehensive(item_nm_alias_values)
    print(f"selected by select_most_comprehensive: {selected}")

    # Check if '아이폰 17' (the alias_from for 아이폰17) is selected
    print(f"\nIs '아이폰 17' in selected? {'아이폰 17' in selected}")

    # Final filter
    adf = pd.DataFrame([{'item_nm': k, 'item_nm_alias': v} for k, v in result_dict.items()])
    result_aliases = list(adf.query("item_nm_alias in @selected")['item_nm'].unique())

    print(f"\nFinal aliases ({len(result_aliases)}):")
    for alias in result_aliases:
        marker = " <-- TARGET" if alias == '아이폰17' else ""
        print(f"  '{alias}'{marker}")


if __name__ == '__main__':
    debug_missing_alias()
