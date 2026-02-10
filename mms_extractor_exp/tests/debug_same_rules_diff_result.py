"""
Debug why same rules produce different results for two CSVs.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd
from services.item_data_loader import ItemDataLoader


def generate_aliases(item_nm: str, alias_pdf) -> list:
    """Generate aliases for a given item_nm using provided alias rules."""
    loader = ItemDataLoader(data_source='local')

    test_df = pd.DataFrame([{
        'item_nm': item_nm,
        'item_id': 'TEST',
        'item_desc': item_nm,
        'item_dmn': 'E'
    }])

    result_df = loader.apply_cascading_alias_rules(test_df, alias_pdf, max_depth=3)
    return result_df['item_nm_alias'].tolist()


def debug_same_rules():
    """Show that the difference is in INPUT, not RULES."""

    # Load and prepare alias rules (same for both)
    loader = ItemDataLoader(data_source='local')
    alias_pdf = loader.load_alias_rules()

    # Use a dummy df for expand_build_aliases
    dummy_df = pd.DataFrame([{'item_nm': 'dummy', 'item_desc': 'dummy', 'item_dmn': 'X'}])
    alias_pdf = loader.expand_build_aliases(alias_pdf, dummy_df)
    alias_pdf = loader.create_bidirectional_aliases(alias_pdf)

    print("=" * 70)
    print("TEST 1: Same rules, same input → Same result")
    print("=" * 70)

    aliases_a = generate_aliases('iPhone 17', alias_pdf)
    aliases_b = generate_aliases('iPhone 17', alias_pdf)

    print(f"\nInput: 'iPhone 17' (both cases)")
    print(f"Result A: {len(aliases_a)} aliases")
    print(f"Result B: {len(aliases_b)} aliases")
    print(f"Are they equal? {set(aliases_a) == set(aliases_b)}")

    print("\n" + "=" * 70)
    print("TEST 2: Same rules, DIFFERENT input → DIFFERENT result")
    print("=" * 70)

    aliases_iphone17 = generate_aliases('iPhone 17', alias_pdf)
    aliases_iphone17promax = generate_aliases('IPHONE 17 PRO MAX', alias_pdf)

    print(f"\nInput A: 'iPhone 17'")
    print(f"  → {len(aliases_iphone17)} aliases")

    print(f"\nInput B: 'IPHONE 17 PRO MAX'")
    print(f"  → {len(aliases_iphone17promax)} aliases")

    print(f"\nDifference: {len(aliases_iphone17promax) - len(aliases_iphone17)} more aliases")

    print("\n" + "=" * 70)
    print("THE REAL ISSUE: generate_alias_list.py picks different inputs")
    print("=" * 70)
    print("""
In generate_alias_list.py, when you run:
  python generate_alias_list.py "iPhone 17" --csv <csv_file>

The script searches the CSV for the input string:

1. Original CSV (offer_master_data.csv):
   - Searches for 'iPhone 17'
   - EXACT MATCH found: 'iPhone 17'
   - Uses 'iPhone 17' as input → 12 aliases

2. Rev CSV (offer_master_data_251023_rev.csv):
   - Searches for 'iPhone 17'
   - NO exact match (items are uppercase: 'IPHONE 17')
   - PARTIAL MATCH found: 'IPHONE 17 PRO MAX'
   - Uses 'IPHONE 17 PRO MAX' as input → 25 aliases

The RULES are identical. The INPUT is different!
""")

    print("=" * 70)
    print("VERIFICATION: Using same input with both CSVs")
    print("=" * 70)

    # Force same input regardless of CSV
    print("\nForcing input = 'iPhone 17' for both:")
    a1 = generate_aliases('iPhone 17', alias_pdf)
    print(f"  Result: {len(a1)} aliases")

    print("\nForcing input = 'IPHONE 17 PRO MAX' for both:")
    a2 = generate_aliases('IPHONE 17 PRO MAX', alias_pdf)
    print(f"  Result: {len(a2)} aliases")

    print("\n→ Same input = Same output (rules are identical)")


if __name__ == '__main__':
    debug_same_rules()
