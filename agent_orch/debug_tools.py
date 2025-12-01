import os
import sys
import logging
import pandas as pd
# Add workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_orch.tools import EntitySearchTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_search():
    # Path to CSV
    csv_path = "/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/data/offer_master_data.csv"
    
    print(f"Initializing EntitySearchTool with {csv_path}...")
    tool = EntitySearchTool(csv_path)
    
    # Inspect DataFrame
    if tool.df is None or tool.df.empty:
        print("ERROR: DataFrame is empty or None!")
    else:
        print(f"DataFrame loaded with {len(tool.df)} rows.")
        print(f"Columns: {tool.df.columns.tolist()}")
        print("First 5 'ITEM_NM' values:")
        print(tool.df['ITEM_NM'].head().tolist())
        
        # Check for iPhone items (English)
        iphone_items = tool.df[tool.df['ITEM_NM'].astype(str).str.contains('iPhone', case=False, na=False)]
        print(f"\nFound {len(iphone_items)} items containing 'iPhone' in ITEM_NM.")
        
        # Check for iPhone items (Korean)
        korean_iphone_items = tool.df[tool.df['ITEM_NM'].astype(str).str.contains('아이폰', case=False, na=False)]
        print(f"Found {len(korean_iphone_items)} items containing '아이폰' in ITEM_NM.")
        if not korean_iphone_items.empty:
            print("Sample Korean items:", korean_iphone_items['ITEM_NM'].head().tolist())

        # Check ITEM_ALS
        if 'ITEM_ALS' in tool.df.columns:
            print("\nChecking ITEM_ALS column...")
            alias_items = tool.df[tool.df['ITEM_ALS'].astype(str).str.contains('iPhone', case=False, na=False)]
            print(f"Found {len(alias_items)} items containing 'iPhone' in ITEM_ALS.")

    # Test Search
    queries = ["아이폰"]
    for q in queries:
        print(f"\nTesting search with query: '{q}' and limit=20...")
        result = tool.search(q, limit=20, threshold=0)
        print(f"Result for '{q}':")
        print(result)

if __name__ == "__main__":
    debug_search()
