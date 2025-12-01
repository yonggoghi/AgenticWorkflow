"""
Data loader for MMS Agent
Loads necessary data files independently
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads data from CSV files in mms_extractor_exp/data directory
    This is a simplified, independent version
    """
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Default to mms_extractor_exp/data
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.data_dir = os.path.join(base_dir, "mms_extractor_exp", "data")
        else:
            self.data_dir = data_dir
        
        self.item_pdf = None
        self.org_pdf = None
        self.alias_pdf = None
        self.pgm_pdf = None
        
    def load_item_data(self, file_name="offer_master_data.csv"):
        """Load item/product data"""
        try:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Item data file not found: {file_path}")
                self.item_pdf = pd.DataFrame()
                return
            
            self.item_pdf = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # Normalize column names to lowercase
            self.item_pdf.columns = self.item_pdf.columns.str.lower()
            
            # Create item_nm_alias if not exists
            if 'item_nm_alias' not in self.item_pdf.columns:
                if 'item_als' in self.item_pdf.columns:
                    self.item_pdf['item_nm_alias'] = self.item_pdf['item_als']
                elif 'item_nm' in self.item_pdf.columns:
                    self.item_pdf['item_nm_alias'] = self.item_pdf['item_nm']
            
            logger.info(f"Loaded {len(self.item_pdf)} items")
        except Exception as e:
            logger.error(f"Failed to load item data: {e}")
            self.item_pdf = pd.DataFrame()
    
    def load_org_data(self, file_name="org_info_all_250605.csv"):
        """Load organization/store data"""
        try:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Org data file not found: {file_path}")
                self.org_pdf = pd.DataFrame()
                return
            
            # Try different encodings
            for encoding in ['cp949', 'euc-kr', 'utf-8-sig', 'utf-8']:
                try:
                    self.org_pdf = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Org data loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error("Failed to load org data with any encoding")
                self.org_pdf = pd.DataFrame()
                return
            
            # Normalize column names
            self.org_pdf.columns = self.org_pdf.columns.str.lower()
            
            # Rename columns for consistency
            if 'org_nm' in self.org_pdf.columns and 'item_nm' not in self.org_pdf.columns:
                self.org_pdf['item_nm'] = self.org_pdf['org_nm']
            if 'org_cd' in self.org_pdf.columns and 'item_id' not in self.org_pdf.columns:
                self.org_pdf['item_id'] = self.org_pdf['org_cd']
            
            logger.info(f"Loaded {len(self.org_pdf)} organizations")
        except Exception as e:
            logger.error(f"Failed to load org data: {e}")
            self.org_pdf = pd.DataFrame()
    
    def load_alias_data(self, file_name="alias_rules.csv"):
        """Load alias rules"""
        try:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Alias data file not found: {file_path}")
                self.alias_pdf = pd.DataFrame()
                return
            
            self.alias_pdf = pd.read_csv(file_path, encoding='utf-8-sig')
            # Normalize column names
            self.alias_pdf.columns = self.alias_pdf.columns.str.lower()
            logger.info(f"Loaded {len(self.alias_pdf)} alias rules")
        except Exception as e:
            logger.error(f"Failed to load alias data: {e}")
            self.alias_pdf = pd.DataFrame()
    
    def load_program_data(self, file_name="pgm_tag_ext_250516.csv"):
        """Load program classification data"""
        try:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Program data file not found: {file_path}")
                self.pgm_pdf = pd.DataFrame()
                return
            
            self.pgm_pdf = pd.read_csv(file_path, encoding='utf-8-sig')
            # Normalize column names
            self.pgm_pdf.columns = self.pgm_pdf.columns.str.lower()
            logger.info(f"Loaded {len(self.pgm_pdf)} programs")
        except Exception as e:
            logger.error(f"Failed to load program data: {e}")
            self.pgm_pdf = pd.DataFrame()
    
    def load_all(self):
        """Load all data files"""
        self.load_item_data()
        self.load_org_data()
        self.load_alias_data()
        self.load_program_data()
        
        return self
