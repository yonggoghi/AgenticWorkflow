"""
Debug script to investigate why exact alias rules are not being applied.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd


def debug_alias_matching():
    """Debug why exact rules might not match."""

    # 1. Load alias rules
    alias_pdf = pd.read_csv('./data/alias_rules.csv')

    # 2. Find the exact rule for 아이폰 17
    print("=" * 60)
    print("1. Finding '아이폰 17' exact rules in CSV")
    print("=" * 60)

    exact_rules = alias_pdf[alias_pdf['type'] == 'exact']
    iphone17_rules = exact_rules[exact_rules['alias_1'].str.contains('아이폰 17', na=False)]
    print(f"Found {len(iphone17_rules)} rules containing '아이폰 17':")
    print(iphone17_rules[['alias_1', 'alias_2', 'type']])

    # 3. Check the exact string from CSV
    print("\n" + "=" * 60)
    print("2. Examining string bytes")
    print("=" * 60)

    # Get alias_1 value from CSV (after explode)
    alias_pdf_exploded = alias_pdf.copy()
    alias_pdf_exploded['alias_1'] = alias_pdf_exploded['alias_1'].str.split("&&")
    alias_pdf_exploded = alias_pdf_exploded.explode('alias_1')

    csv_string = alias_pdf_exploded[
        (alias_pdf_exploded['alias_1'] == '아이폰 17') &
        (alias_pdf_exploded['type'] == 'exact')
    ]['alias_1'].iloc[0] if len(alias_pdf_exploded[
        (alias_pdf_exploded['alias_1'] == '아이폰 17') &
        (alias_pdf_exploded['type'] == 'exact')
    ]) > 0 else None

    # Simulated string from partial rule replacement
    generated_string = 'iPhone 17'.replace('iPhone', '아이폰')

    print(f"\nCSV string:       '{csv_string}'")
    print(f"Generated string: '{generated_string}'")
    print(f"Are they equal?   {csv_string == generated_string}")

    if csv_string:
        print(f"\nCSV string bytes:       {[hex(ord(c)) for c in csv_string]}")
        print(f"Generated string bytes: {[hex(ord(c)) for c in generated_string]}")

        print(f"\nCSV string repr:       {repr(csv_string)}")
        print(f"Generated string repr: {repr(generated_string)}")

        # Find differences
        if csv_string != generated_string:
            print("\nDifferences found:")
            max_len = max(len(csv_string), len(generated_string))
            for i in range(max_len):
                csv_char = csv_string[i] if i < len(csv_string) else '<missing>'
                gen_char = generated_string[i] if i < len(generated_string) else '<missing>'
                if csv_char != gen_char:
                    print(f"  Position {i}: CSV='{csv_char}' ({hex(ord(csv_char)) if csv_char != '<missing>' else 'N/A'}) vs Generated='{gen_char}' ({hex(ord(gen_char)) if gen_char != '<missing>' else 'N/A'})")

    # 4. Check all alias_1 values that should match '아이폰 17'
    print("\n" + "=" * 60)
    print("3. All exploded alias_1 values for exact type")
    print("=" * 60)

    exact_exploded = alias_pdf_exploded[alias_pdf_exploded['type'] == 'exact']
    matching = exact_exploded[exact_exploded['alias_1'].str.strip() == '아이폰 17']
    print(f"\nRules where alias_1.strip() == '아이폰 17': {len(matching)}")
    if not matching.empty:
        print(matching[['alias_1', 'alias_2', 'type']])
        for idx, row in matching.iterrows():
            print(f"\n  alias_1 repr: {repr(row['alias_1'])}")
            print(f"  Has leading/trailing whitespace: {row['alias_1'] != row['alias_1'].strip()}")


if __name__ == '__main__':
    os.chdir(project_root)
    debug_alias_matching()
