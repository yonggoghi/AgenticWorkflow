"""
Debug why iPhone 17 doesn't generate iPhone 17 Pro even though both exist in CSV.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd


def debug_both_csv():
    """Show that both CSVs have the same items but alias generation differs."""

    print("=" * 70)
    print("1. Items in both CSVs")
    print("=" * 70)

    df1 = pd.read_csv("data/offer_master_data.csv")
    df2 = pd.read_csv("data/offer_master_data_251023_rev.csv")

    print("\nOriginal CSV (ITEM_NM):")
    items1 = df1.query("ITEM_NM.str.contains('iPhone 17', case=False, na=False) and ITEM_DMN=='E'")[['ITEM_NM']].drop_duplicates()
    print(items1.to_string(index=False))

    print("\nRev CSV (ITEM_NM from ITEM_DESC):")
    items2 = df2.query("ITEM_NM.str.contains('IPHONE 17', case=False, na=False) and ITEM_DMN=='E'")[['ITEM_NM']].drop_duplicates()
    print(items2.to_string(index=False))

    print("\n" + "=" * 70)
    print("2. The Problem")
    print("=" * 70)
    print("""
When we search for 'iPhone 17' in generate_alias_list.py:

Original CSV:
  - Exact match: 'iPhone 17' found
  - Uses 'iPhone 17' as the item to generate aliases for
  - Result: 12 aliases (no 'iPhone 17 Pro')

Rev CSV:
  - No exact match for 'iPhone 17' (items are uppercase: 'IPHONE 17')
  - Partial match finds: 'IPHONE 17 PRO MAX'
  - Uses 'IPHONE 17 PRO MAX' as the item
  - Result: 25 aliases
""")

    print("=" * 70)
    print("3. Why 'iPhone 17 Pro' is NOT generated from 'iPhone 17'")
    print("=" * 70)
    print("""
The alias rules (lines 185-186) define:

  alias_1: 아이폰 17, 아이폰 17 PRO, 아이폰 17 PRO MAX
  alias_2: 아이폰17, 아이폰 17, 아이폰 17 시리즈
  direction: U (UNIDIRECTIONAL)
  type: exact

This creates ONE-WAY mappings:
  '아이폰 17 PRO' → '아이폰 17' ✓  (PRO to base)
  '아이폰 17 PRO MAX' → '아이폰 17' ✓  (PRO MAX to base)
  '아이폰 17' → '아이폰 17 PRO' ✗  (base to PRO - NOT DEFINED!)

The rules are designed to NORMALIZE variants to a base form,
NOT to EXPAND a base form to its variants.
""")

    print("=" * 70)
    print("4. Solution Options")
    print("=" * 70)
    print("""
Option A: Change direction='U' to direction='B' (bidirectional)
  - This would create reverse mappings automatically
  - '아이폰 17' → '아이폰 17 PRO' would be generated

Option B: Add explicit reverse rules
  - Add new rule: '아이폰 17' → '아이폰 17 PRO&&아이폰 17 PRO MAX'

Option C: Use 'build' type rule
  - Create a 'build' type rule that dynamically finds items in CSV
  - e.g., alias_1='아이폰 17', type='build' would find all items containing '아이폰 17'
""")

    # Check if there are any 'build' type rules
    print("=" * 70)
    print("5. Current 'build' type rules in alias_rules.csv")
    print("=" * 70)

    alias_pdf = pd.read_csv('./data/alias_rules.csv')
    build_rules = alias_pdf[alias_pdf['type'] == 'build']
    print(f"\nFound {len(build_rules)} 'build' type rules:")
    if len(build_rules) > 0:
        print(build_rules[['alias_1', 'alias_2', 'type']].to_string())
    else:
        print("  None!")


if __name__ == '__main__':
    debug_both_csv()
