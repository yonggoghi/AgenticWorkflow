import os
import sys
import logging
import pandas as pd
from rapidfuzz import process, fuzz
from typing import List, Dict, Any

# Ensure mms_extractor_exp is in path for imports
mms_exp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mms_extractor_exp")
if mms_exp_path not in sys.path:
    sys.path.insert(0, mms_exp_path)

# Mock torch and kiwipiepy if not present, as we only need data loading logic
try:
    import torch
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['torch'] = MagicMock()
    sys.modules['torch.backends'] = MagicMock()
    sys.modules['torch.backends.mps'] = MagicMock()
    sys.modules['torch.cuda'] = MagicMock()

try:
    import kiwipiepy
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['kiwipiepy'] = MagicMock()

try:
    import cx_Oracle
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['cx_Oracle'] = MagicMock()

try:
    import joblib
except ImportError:
    # Functional mock for joblib to handle Parallel execution
    class MockParallel:
        def __init__(self, n_jobs=1, backend='threading'):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def __call__(self, iterable):
            # Just execute the generator/list
            return list(iterable)

    def mock_delayed(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    from unittest.mock import MagicMock
    sys.modules['joblib'] = MagicMock()
    sys.modules['joblib'].Parallel = MockParallel
    sys.modules['joblib'].delayed = mock_delayed

try:
    import sentence_transformers
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['sentence_transformers'] = MagicMock()

try:
    import networkx
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['networkx'] = MagicMock()

try:
    import graphviz
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['graphviz'] = MagicMock()

try:
    from mms_extractor_data import MMSExtractorDataMixin
    from config.settings import METADATA_CONFIG, PROCESSING_CONFIG
except ImportError as e:
    logging.error(f"Failed to import from mms_extractor_exp: {e}")
    # Define dummy mixin if import fails to avoid crash during editing
    class MMSExtractorDataMixin:
        pass

logger = logging.getLogger(__name__)

class EntitySearchTool(MMSExtractorDataMixin):
    def __init__(self, csv_path: str = None):
        """
        Initialize the search tool using MMSExtractorDataMixin logic.
        Args:
            csv_path: Optional path to override the default data path. 
                      If provided, it updates METADATA_CONFIG.offer_data_path temporarily.
        """
        # Set up attributes required by MMSExtractorDataMixin
        self.offer_info_data_src = "local"
        self.data_dir = os.path.join(mms_exp_path, "data")
        
        # Override config path if csv_path is provided
        if csv_path:
            # We need to ensure we don't permanently mutate the global config if possible,
            # but for this tool instance, we want to use the provided path.
            # Since METADATA_CONFIG is a dataclass instance, we can set it.
            # However, the mixin uses getattr(METADATA_CONFIG, ...).
            # Let's just set the attribute on self if the mixin supports it, 
            # but the mixin reads from METADATA_CONFIG directly in _load_and_prepare_item_data.
            # A better way is to mock the config or just rely on the mixin's default if csv_path matches.
            pass

        # Initialize data
        try:
            # We need to make sure METADATA_CONFIG points to the correct files
            # The mixin uses relative paths like './data/alias_rules.csv' which might be wrong
            # if running from a different CWD.
            # We should update the paths in METADATA_CONFIG to be absolute if they are not.
            
            # Fix paths in METADATA_CONFIG to be absolute based on mms_exp_path
            if hasattr(METADATA_CONFIG, 'offer_data_path'):
                if not os.path.isabs(METADATA_CONFIG.offer_data_path):
                    METADATA_CONFIG.offer_data_path = os.path.join(mms_exp_path, METADATA_CONFIG.offer_data_path.lstrip("./"))
            
            if hasattr(METADATA_CONFIG, 'alias_rules_path'):
                if not os.path.isabs(METADATA_CONFIG.alias_rules_path):
                    METADATA_CONFIG.alias_rules_path = os.path.join(mms_exp_path, METADATA_CONFIG.alias_rules_path.lstrip("./"))

            # If csv_path was passed, use it
            if csv_path:
                 METADATA_CONFIG.offer_data_path = csv_path

            self._load_and_prepare_item_data()
            self.df = self.item_pdf_all
            logger.info(f"EntitySearchTool initialized with {len(self.df)} items.")
            
        except Exception as e:
            logger.error(f"Failed to initialize EntitySearchTool: {e}")
            self.df = pd.DataFrame()

    def search(self, query: str, limit: int = 10, threshold: float = 50.0) -> List[Dict[str, Any]]:
        """
        Search for entities matching the query.
        Returns a list of dictionaries containing top K results.
        """
        if self.df is None or self.df.empty:
            return []

        if not query:
            return []

        # Prepare choices for rapidfuzz
        # Use item_nm_alias if available, otherwise item_nm
        if 'item_nm_alias' in self.df.columns:
            choices = self.df['item_nm_alias'].astype(str).tolist()
        else:
            choices = self.df['item_nm'].astype(str).tolist()
        
        # Fuzzy search
        results = process.extract(
            query, 
            choices, 
            scorer=fuzz.WRatio, 
            limit=limit,
            score_cutoff=threshold
        )

        if not results:
            return []

        # Format results
        output = []
        seen_ids = set()
        
        for name, score, index in results:
            row = self.df.iloc[index]
            item_id = row.get('item_id', 'N/A')
            item_nm = row.get('item_nm', 'N/A')
            
            # Deduplicate by ID to avoid showing same item multiple times due to different aliases
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            
            output.append({
                "item_nm": item_nm,
                "item_id": item_id,
                "score": score,
                "matched_alias": name
            })

        return output

if __name__ == "__main__":
    # Test
    tool = EntitySearchTool()
    print(tool.search("아이폰"))
