"""
Save ItemDataLoader DataFrame to tests directory.

This script loads item data using ItemDataLoader and saves the resulting
DataFrames to CSV files for testing and inspection purposes.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from services.item_data_loader import ItemDataLoader


def save_item_dataframes(output_dir: str = None):
    """
    Load item data and save to CSV files.

    Args:
        output_dir: Directory to save the CSV files. Defaults to tests/item_data_output/
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'item_data_output')

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ItemDataLoader DataFrame Save Script")
    print("=" * 60)

    # Initialize loader
    loader = ItemDataLoader(data_source='local')

    # Load and prepare items
    print("\nLoading and preparing item data...")
    item_df, alias_df = loader.load_and_prepare_items()

    # Save item DataFrame
    item_output_path = os.path.join(output_dir, 'item_df.csv')
    item_df.to_csv(item_output_path, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] Item DataFrame: {item_output_path}")
    print(f"  - Shape: {item_df.shape}")
    print(f"  - Columns: {list(item_df.columns)}")

    # Save alias DataFrame
    alias_output_path = os.path.join(output_dir, 'alias_df.csv')
    alias_df.to_csv(alias_output_path, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] Alias DataFrame: {alias_output_path}")
    print(f"  - Shape: {alias_df.shape}")
    print(f"  - Columns: {list(alias_df.columns)}")

    # Save raw alias DataFrame (before processing)
    if loader.alias_pdf_raw is not None:
        raw_alias_path = os.path.join(output_dir, 'alias_raw_df.csv')
        loader.alias_pdf_raw.to_csv(raw_alias_path, index=False, encoding='utf-8-sig')
        print(f"\n[Saved] Raw Alias DataFrame: {raw_alias_path}")
        print(f"  - Shape: {loader.alias_pdf_raw.shape}")

    # Print sample data
    print("\n" + "=" * 60)
    print("Sample Data Preview")
    print("=" * 60)

    if not item_df.empty:
        print("\nItem DataFrame (first 5 rows):")
        print(item_df[['item_nm', 'item_id', 'item_nm_alias', 'item_dmn']].head())

    if not alias_df.empty:
        print("\nAlias DataFrame (first 5 rows):")
        print(alias_df.head())

    print("\n" + "=" * 60)
    print(f"All files saved to: {output_dir}")
    print("=" * 60)

    return item_df, alias_df


if __name__ == '__main__':
    save_item_dataframes("tests/item_data_output_4")
