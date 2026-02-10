"""
Debug expand_build_aliases effect on the alias rules.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd
from services.item_data_loader import ItemDataLoader


def debug_expand_build():
    """Check if expand_build_aliases removes the exact rule."""

    loader = ItemDataLoader(data_source='local')

    # Step 1: Load alias rules (before expand_build)
    alias_pdf_before = loader.load_alias_rules()

    print("=" * 60)
    print("1. After load_alias_rules (before expand_build)")
    print("=" * 60)

    target_before = alias_pdf_before[
        (alias_pdf_before['alias_1'] == '아이폰 17') &
        (alias_pdf_before['alias_2'] == '아이폰17')
    ]
    print(f"Rule '아이폰 17' → '아이폰17' exists: {len(target_before) > 0}")
    if len(target_before) > 0:
        print(target_before[['alias_1', 'alias_2', 'type', 'direction']])

    # Step 2: Create minimal test df
    test_df = pd.DataFrame([{
        'item_nm': 'iPhone 17',
        'item_id': 'TEST_ID',
        'item_desc': 'iPhone 17',
        'item_dmn': 'E'
    }])

    # Step 3: Expand build aliases
    alias_pdf_after_build = loader.expand_build_aliases(alias_pdf_before, test_df)

    print("\n" + "=" * 60)
    print("2. After expand_build_aliases")
    print("=" * 60)

    target_after_build = alias_pdf_after_build[
        (alias_pdf_after_build['alias_1'] == '아이폰 17') &
        (alias_pdf_after_build['alias_2'] == '아이폰17')
    ]
    print(f"Rule '아이폰 17' → '아이폰17' exists: {len(target_after_build) > 0}")
    if len(target_after_build) > 0:
        print(target_after_build[['alias_1', 'alias_2', 'type', 'direction']])

    # Step 4: Create bidirectional aliases
    alias_pdf_final = loader.create_bidirectional_aliases(alias_pdf_after_build)

    print("\n" + "=" * 60)
    print("3. After create_bidirectional_aliases")
    print("=" * 60)

    target_final = alias_pdf_final[
        (alias_pdf_final['alias_1'] == '아이폰 17') &
        (alias_pdf_final['alias_2'] == '아이폰17')
    ]
    print(f"Rule '아이폰 17' → '아이폰17' exists: {len(target_final) > 0}")
    if len(target_final) > 0:
        print(target_final[['alias_1', 'alias_2', 'type', 'direction']])

    # Check total rules with alias_1='아이폰 17'
    print("\n" + "=" * 60)
    print("4. All rules with alias_1='아이폰 17'")
    print("=" * 60)

    all_target = alias_pdf_final[alias_pdf_final['alias_1'] == '아이폰 17']
    print(f"Total {len(all_target)} rules:")
    print(all_target[['alias_1', 'alias_2', 'type', 'direction']])


if __name__ == '__main__':
    debug_expand_build()
