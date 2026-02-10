"""
Apply ITEM_NM rule and save revised CSV.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import pandas as pd

# Load CSV
input_path = 'data/offer_master_data_251023.csv'
output_path = 'data/offer_master_data_251023_rev.csv'

print(f"Loading {input_path}...")
df = pd.read_csv(input_path)
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Check before
e_count = len(df[df['ITEM_DMN'] == 'E'])
print(f"\nRows with ITEM_DMN='E': {e_count}")

# Apply rule
df['ITEM_NM'] = df.apply(
    lambda x: x['ITEM_DESC'] if x['ITEM_DMN'] == 'E' and pd.notna(x['ITEM_DESC']) else x['ITEM_NM'],
    axis=1
)

print("Applied rule: ITEM_NM = ITEM_DESC when ITEM_DMN='E' and ITEM_DESC is not null")

# Save
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nSaved to {output_path}")
