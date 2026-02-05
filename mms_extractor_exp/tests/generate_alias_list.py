"""
Generate alias list for a given string using ItemDataLoader.

Usage:
    ../venv/bin/python tests/generate_alias_list.py "iPhone 17"
    ../venv/bin/python tests/generate_alias_list.py "갤럭시 S25"
    ../venv/bin/python tests/generate_alias_list.py "iPhone 17" --csv data/offer_master_data_251023_rev.csv
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd
from services.item_data_loader import ItemDataLoader


def generate_alias_list(item_nm: str, offer_csv: str = None) -> list:
    """
    Generate alias list for a given string.

    Args:
        item_nm: The item name to generate aliases for
        offer_csv: Optional path to offer CSV file for build aliases

    Returns:
        List of alias strings
    """
    loader = ItemDataLoader(data_source='local')
    alias_pdf = loader.load_alias_rules()

    # Load item_df from CSV for expand_build_aliases, or use test item
    if offer_csv:
        item_df = pd.read_csv(offer_csv)
        item_df.columns = [c.lower() for c in item_df.columns]
        print(f"Loaded {len(item_df)} items from {offer_csv}")
    else:
        item_df = pd.DataFrame([{
            'item_nm': item_nm,
            'item_id': 'TEST',
            'item_desc': item_nm,
            'item_dmn': 'E'
        }])

    # Create test_df with the input item_nm
    test_df = pd.DataFrame([{
        'item_nm': item_nm,
        'item_id': 'TEST',
        'item_desc': item_nm,
        'item_dmn': 'E'
    }])

    alias_pdf = loader.expand_build_aliases(alias_pdf, item_df)
    alias_pdf = loader.create_bidirectional_aliases(alias_pdf)

    result_df = loader.apply_cascading_alias_rules(test_df, alias_pdf, max_depth=3)
    return result_df['item_nm_alias'].tolist()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_alias_list.py <item_name> [--csv <path>]")
        print("Example: python generate_alias_list.py 'iPhone 17'")
        print("Example: python generate_alias_list.py 'iPhone 17' --csv data/offer_master_data_251023_rev.csv")
        sys.exit(1)

    item_nm = sys.argv[1]
    offer_csv = None

    if '--csv' in sys.argv:
        csv_idx = sys.argv.index('--csv')
        if csv_idx + 1 < len(sys.argv):
            offer_csv = sys.argv[csv_idx + 1]

    aliases = generate_alias_list(item_nm, offer_csv)

    print(f"\nAliases for '{item_nm}' ({len(aliases)}):")
    for alias in aliases:
        print(f"  - {alias}")
