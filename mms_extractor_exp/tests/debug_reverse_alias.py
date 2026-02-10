"""
Debug why 'iPhone 17' doesn't generate 'iPhone 17 Pro' and 'iPhone 17 Pro Max'.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd


def debug_reverse_alias():
    """Check why reverse aliases are not generated."""

    alias_pdf = pd.read_csv('./data/alias_rules.csv')

    # Show line 185-187 raw data
    print("=" * 70)
    print("Raw alias rules (lines 183-187 in exploded form):")
    print("=" * 70)

    # Filter rows containing '아이폰 17'
    iphone17_rules = alias_pdf[alias_pdf['alias_1'].str.contains('아이폰 17', na=False)]
    print(iphone17_rules.to_string())

    # Explode and check
    alias_pdf_exp = alias_pdf.copy()
    alias_pdf_exp['alias_1'] = alias_pdf_exp['alias_1'].str.split("&&")
    alias_pdf_exp['alias_2'] = alias_pdf_exp['alias_2'].str.split("&&")
    alias_pdf_exp = alias_pdf_exp.explode('alias_1')
    alias_pdf_exp = alias_pdf_exp.explode('alias_2')

    print("\n" + "=" * 70)
    print("Question: Does a rule exist to map '아이폰 17' → '아이폰 17 PRO'?")
    print("=" * 70)

    # Check if there's a rule: 아이폰 17 → 아이폰 17 PRO
    forward_rule = alias_pdf_exp[
        (alias_pdf_exp['alias_1'] == '아이폰 17') &
        (alias_pdf_exp['alias_2'].str.contains('PRO', na=False))
    ]
    print(f"\nRules '아이폰 17' → '*PRO*': {len(forward_rule)}")
    if len(forward_rule) > 0:
        print(forward_rule[['alias_1', 'alias_2', 'direction', 'type']])
    else:
        print("  None found!")

    print("\n" + "=" * 70)
    print("Current rules for '아이폰 17' as alias_1:")
    print("=" * 70)

    a17_as_from = alias_pdf_exp[alias_pdf_exp['alias_1'] == '아이폰 17']
    print(f"\nFound {len(a17_as_from)} rules:")
    for _, row in a17_as_from.iterrows():
        print(f"  '아이폰 17' → '{row['alias_2']}' (direction={row['direction']}, type={row['type']})")

    print("\n" + "=" * 70)
    print("Current rules for '아이폰 17 PRO' as alias_1:")
    print("=" * 70)

    a17pro_as_from = alias_pdf_exp[alias_pdf_exp['alias_1'] == '아이폰 17 PRO']
    print(f"\nFound {len(a17pro_as_from)} rules:")
    for _, row in a17pro_as_from.iterrows():
        print(f"  '아이폰 17 PRO' → '{row['alias_2']}' (direction={row['direction']}, type={row['type']})")

    print("\n" + "=" * 70)
    print("EXPLANATION:")
    print("=" * 70)
    print("""
The alias rules define:
  - Line 185: 아이폰 17, 아이폰 17 PRO, 아이폰 17 PRO MAX → 아이폰 신제품, 아이폰 최신폰, ...
  - Line 186: 아이폰 17, 아이폰 17 PRO, 아이폰 17 PRO MAX → 아이폰17, 아이폰 17, 아이폰 17 시리즈

These rules are UNIDIRECTIONAL (direction='U'):
  - '아이폰 17 PRO' → '아이폰 17' ✓ (maps PRO version to base)
  - '아이폰 17' → '아이폰 17 PRO' ✗ (reverse NOT defined)

To generate 'iPhone 17 Pro' from 'iPhone 17', you would need:
  1. A BIDIRECTIONAL rule (direction='B'), OR
  2. An explicit reverse rule: '아이폰 17' → '아이폰 17 PRO'
""")

    # Check if bidirectional would help
    print("=" * 70)
    print("If line 186 had direction='B' instead of 'U':")
    print("=" * 70)
    print("""
With direction='B', the system would create reverse rules:
  - '아이폰17' → '아이폰 17'
  - '아이폰17' → '아이폰 17 PRO'
  - '아이폰17' → '아이폰 17 PRO MAX'
  - '아이폰 17' → '아이폰 17'  (self)
  - '아이폰 17' → '아이폰 17 PRO'  ← This would enable the reverse!
  - '아이폰 17' → '아이폰 17 PRO MAX'
  - '아이폰 17 시리즈' → '아이폰 17'
  - '아이폰 17 시리즈' → '아이폰 17 PRO'
  - '아이폰 17 시리즈' → '아이폰 17 PRO MAX'

Then 'iPhone 17' → '아이폰 17' → '아이폰 17 PRO' → 'iPhone 17 PRO' would work!
""")


if __name__ == '__main__':
    debug_reverse_alias()
