"""
Test script to generate alias list for a given string using existing ItemDataLoader.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd
from services.item_data_loader import ItemDataLoader
from utils import select_most_comprehensive


def generate_aliases_for_string_debug(item_nm: str):
    """Generate alias list with detailed debug output."""

    print("=" * 60)
    print(f"DEBUG: Generating aliases for: '{item_nm}'")
    print("=" * 60)

    # Initialize loader and load alias rules
    loader = ItemDataLoader(data_source='local')
    alias_pdf = loader.load_alias_rules()

    test_df = pd.DataFrame([{
        'item_nm': item_nm,
        'item_id': 'TEST_ID',
        'item_desc': item_nm,
        'item_dmn': 'E'
    }])

    alias_pdf = loader.expand_build_aliases(alias_pdf, test_df)
    alias_pdf = loader.create_bidirectional_aliases(alias_pdf)

    # Manually run cascade logic with debug
    alias_rule_set = list(zip(
        alias_pdf['alias_1'],
        alias_pdf['alias_2'],
        alias_pdf['type']
    ))

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

                if new_item not in result_dict:
                    result_dict[new_item] = alias_from.strip()
                    to_process.append((new_item, depth + 1, path_applied_rules | {rule_key}))

    # Check result_dict
    print(f"\nresult_dict has {len(result_dict)} items")
    print(f"'아이폰17' in result_dict: {'아이폰17' in result_dict}")

    if '아이폰17' in result_dict:
        print(f"  result_dict['아이폰17'] = '{result_dict['아이폰17']}'")

    # Apply select_most_comprehensive
    item_nm_list = [{'item_nm': k, 'item_nm_alias': v} for k, v in result_dict.items()]
    adf = pd.DataFrame(item_nm_list)

    alias_values = adf['item_nm_alias'].tolist()
    print(f"\nUnique item_nm_alias values: {set(alias_values)}")

    selected_alias = select_most_comprehensive(alias_values)
    print(f"select_most_comprehensive result: {selected_alias}")

    # Check if '아이폰 17' is filtered out
    if '아이폰 17' in set(alias_values):
        print(f"'아이폰 17' in alias_values: True")
        print(f"'아이폰 17' in selected_alias: {'아이폰 17' in selected_alias}")

    result_aliases = list(adf.query("item_nm_alias in @selected_alias")['item_nm'].unique())
    if item_nm not in result_aliases:
        result_aliases.append(item_nm)

    print(f"\nFinal aliases ({len(result_aliases)}):")
    for alias in result_aliases:
        marker = " <-- TARGET" if alias == '아이폰17' else ""
        print(f"  - '{alias}'{marker}")

    return result_aliases


def generate_aliases_for_string(item_nm: str):
    """Generate alias list for a given string using ItemDataLoader logic."""

    print("=" * 60)
    print(f"Generating aliases for: '{item_nm}'")
    print("=" * 60)

    loader = ItemDataLoader(data_source='local')
    alias_pdf = loader.load_alias_rules()

    test_df = pd.DataFrame([{
        'item_nm': item_nm,
        'item_id': 'TEST_ID',
        'item_desc': item_nm,
        'item_dmn': 'E'
    }])

    alias_pdf = loader.expand_build_aliases(alias_pdf, test_df)
    alias_pdf = loader.create_bidirectional_aliases(alias_pdf)

    result_df = loader.apply_cascading_alias_rules(test_df, alias_pdf, max_depth=3)
    aliases = result_df['item_nm_alias'].tolist()

    print(f"\nGenerated {len(aliases)} aliases:")
    for alias in aliases:
        marker = " <-- TARGET" if alias == '아이폰17' else ""
        print(f"  - '{alias}'{marker}")

    return aliases


if __name__ == '__main__':
    test_string = 'iPhone 17'
    if len(sys.argv) > 1:
        test_string = sys.argv[1]

    print("\n### Using ItemDataLoader.apply_cascading_alias_rules ###")
    aliases1 = generate_aliases_for_string(test_string)

    print("\n\n### Using manual debug cascade ###")
    aliases2 = generate_aliases_for_string_debug(test_string)

    print("\n\n### Comparison ###")
    print(f"ItemDataLoader: {len(aliases1)} aliases, '아이폰17' present: {'아이폰17' in aliases1}")
    print(f"Manual debug:   {len(aliases2)} aliases, '아이폰17' present: {'아이폰17' in aliases2}")
