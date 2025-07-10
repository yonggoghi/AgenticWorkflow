"""
Data management module for loading and preprocessing data.
"""
import os
import pandas as pd
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

from ..models.language_models import EmbeddingManager
from ..utils.text_processing import preprocess_text
from ..config.settings import DATA_CONFIG, PROCESSING_CONFIG, MODEL_CONFIG


logger = logging.getLogger(__name__)


class DataManager:
    """Manages all data loading and preprocessing operations."""
    
    def __init__(self, use_mock_data: bool = False):
        """
        Initialize data manager.
        
        Args:
            use_mock_data: Whether to use mock data instead of real files
        """
        self.use_mock_data = use_mock_data
        self.item_df = None
        self.pgm_df = None
        self.org_df = None
        self.mms_df = None
        self.stop_words = []
        self.entity_vocab = []
        self.alias_rules = []
        
        # Embeddings
        self.clue_embeddings = None
        self.item_embeddings = None
        self.org_embeddings = None
        
        self.embedding_manager = None
        
        logger.info(f"Initialized data manager (mock_data={use_mock_data})")
    
    def _create_mock_item_data(self) -> pd.DataFrame:
        """Create mock item data for demonstration."""
        mock_data = [
            {
                'item_nm': '베어유',
                'item_id': 'BEAR001',
                'item_desc': '베어유 온라인 강의 서비스',
                'domain': 'education',
                'start_dt': 20250101,
                'end_dt': 99991231,
                'rank': 1,
                'item_nm_alias': '베어유'
            },
            {
                'item_nm': 'T 멤버십',
                'item_id': 'TMEM001',
                'item_desc': 'T 멤버십 서비스',
                'domain': 'membership',
                'start_dt': 20250101,
                'end_dt': 99991231,
                'rank': 1,
                'item_nm_alias': 'T 멤버십'
            },
            {
                'item_nm': '에이닷',
                'item_id': 'ADOT001',
                'item_desc': '에이닷 AI 서비스',
                'domain': 'ai',
                'start_dt': 20250101,
                'end_dt': 99991231,
                'rank': 1,
                'item_nm_alias': '에이닷'
            },
            {
                'item_nm': '뚜레쥬르',
                'item_id': 'TOUS001',
                'item_desc': '뚜레쥬르 베이커리',
                'domain': 'food',
                'start_dt': 20250101,
                'end_dt': 99991231,
                'rank': 1,
                'item_nm_alias': '뚜레쥬르'
            }
        ]
        
        # Add user-defined entities
        for entity in PROCESSING_CONFIG.user_defined_entities:
            mock_data.append({
                'item_nm': entity,
                'item_id': entity,
                'item_desc': entity,
                'domain': 'user_defined',
                'start_dt': 20250101,
                'end_dt': 99991231,
                'rank': 1,
                'item_nm_alias': entity
            })
        
        return pd.DataFrame(mock_data)
    
    def _create_mock_program_data(self) -> pd.DataFrame:
        """Create mock program data for demonstration."""
        mock_data = [
            {
                'pgm_nm': '[마케팅_Care]사용법/혜택/이벤트안내_T멤버십',
                'pgm_id': '2019CCEPM01',
                'clue_tag': 'T멤버십 혜택 안내 이벤트 쿠폰'
            },
            {
                'pgm_nm': '[마케팅_Care]사용법/혜택/이벤트안내_AI',
                'pgm_id': '2025ABCD02',
                'clue_tag': 'AI 서비스 안내 인공지능'
            },
            {
                'pgm_nm': '[마케팅_Care]교육서비스_온라인강의',
                'pgm_id': '2025EDU01',
                'clue_tag': '온라인 강의 교육 학습'
            }
        ]
        
        return pd.DataFrame(mock_data)
    
    def _create_mock_organization_data(self) -> pd.DataFrame:
        """Create mock organization data for demonstration."""
        mock_data = [
            {
                'org_cd': 'D123456789',
                'sub_org_cd': '0001',
                'org_nm': 'SK텔레콤 강남점',
                'org_abbr_nm': 'SKT강남점',
                'bas_addr': '서울시 강남구',
                'dtl_addr': '테헤란로 123'
            },
            {
                'org_cd': 'D987654321',
                'sub_org_cd': '0002',
                'org_nm': 'SK텔레콤 홍대점',
                'org_abbr_nm': 'SKT홍대점',
                'bas_addr': '서울시 마포구',
                'dtl_addr': '홍익로 456'
            }
        ]
        
        return pd.DataFrame(mock_data)
    
    def _create_mock_stop_words(self) -> List[str]:
        """Create mock stop words for demonstration."""
        return ['test', '테스트', 'stop', '중지', 'the', 'a', 'an']
    
    def load_item_data(self) -> pd.DataFrame:
        """
        Load and process item data.
        
        Returns:
            Processed item DataFrame
        """
        logger.info("Loading item data")
        
        if self.use_mock_data or not os.path.exists(DATA_CONFIG.item_info_path):
            logger.info("Using mock item data")
            self.item_df = self._create_mock_item_data()
            self.alias_rules = [('T멤버십', 'T 멤버십'), ('에이닷', '에이닷_자사')]
        else:
            try:
                # Load real item data
                item_df_raw = pd.read_csv(DATA_CONFIG.item_info_path)
                
                # Process item data
                self.item_df = (item_df_raw
                               .drop_duplicates(['item_nm', 'item_id'])
                               [['item_nm', 'item_id', 'item_desc', 'domain', 'start_dt', 'end_dt', 'rank']]
                               .copy())
                
                # Load alias rules
                if os.path.exists(DATA_CONFIG.alias_rules_path):
                    alias_df = pd.read_csv(DATA_CONFIG.alias_rules_path)
                    self.alias_rules = list(zip(alias_df['alias_1'], alias_df['alias_2']))
                else:
                    self.alias_rules = []
                
                # Apply alias rules
                self.item_df['item_nm_alias'] = self.item_df['item_nm'].apply(self._apply_alias_rules)
                self.item_df = self.item_df.explode('item_nm_alias')
                
                # Add user-defined entities
                user_defined_df = pd.DataFrame([
                    {
                        'item_nm': entity,
                        'item_id': entity,
                        'item_desc': entity,
                        'domain': 'user_defined',
                        'start_dt': 20250101,
                        'end_dt': 99991231,
                        'rank': 1,
                        'item_nm_alias': entity
                    }
                    for entity in PROCESSING_CONFIG.user_defined_entities
                ])
                
                self.item_df = pd.concat([self.item_df, user_defined_df], ignore_index=True)
                
            except Exception as e:
                logger.warning(f"Failed to load real item data: {e}. Using mock data instead.")
                self.item_df = self._create_mock_item_data()
                self.alias_rules = [('T멤버십', 'T 멤버십'), ('에이닷', '에이닷_자사')]
        
        logger.info(f"Loaded {len(self.item_df)} item records")
        
        return self.item_df
    
    def load_program_data(self) -> pd.DataFrame:
        """
        Load program data.
        
        Returns:
            Program DataFrame
        """
        logger.info("Loading program data")
        
        if self.use_mock_data or not os.path.exists(DATA_CONFIG.pgm_tag_path):
            logger.info("Using mock program data")
            self.pgm_df = self._create_mock_program_data()
        else:
            try:
                self.pgm_df = pd.read_csv(DATA_CONFIG.pgm_tag_path)
            except Exception as e:
                logger.warning(f"Failed to load real program data: {e}. Using mock data instead.")
                self.pgm_df = self._create_mock_program_data()
        
        logger.info(f"Loaded {len(self.pgm_df)} program records")
        
        return self.pgm_df
    
    def load_organization_data(self) -> pd.DataFrame:
        """
        Load organization data.
        
        Returns:
            Organization DataFrame
        """
        logger.info("Loading organization data")
        
        if self.use_mock_data or not os.path.exists(DATA_CONFIG.org_info_path):
            logger.info("Using mock organization data")
            self.org_df = self._create_mock_organization_data()
        else:
            try:
                self.org_df = pd.read_csv(DATA_CONFIG.org_info_path, encoding='cp949')
                self.org_df['sub_org_cd'] = self.org_df['sub_org_cd'].apply(lambda x: str(x).zfill(4))
            except Exception as e:
                logger.warning(f"Failed to load real organization data: {e}. Using mock data instead.")
                self.org_df = self._create_mock_organization_data()
        
        logger.info(f"Loaded {len(self.org_df)} organization records")
        
        return self.org_df
    
    def load_mms_data(self) -> pd.DataFrame:
        """
        Load MMS data.
        
        Returns:
            MMS DataFrame
        """
        logger.info("Loading MMS data")
        
        if self.use_mock_data or not os.path.exists(DATA_CONFIG.mms_data_path):
            logger.info("Using mock MMS data")
            self.mms_df = pd.DataFrame([
                {
                    'msg_nm': 'Sample MMS 1',
                    'mms_phrs': 'Sample MMS content',
                    'msg': 'Sample MMS 1\nSample MMS content',
                    'offer_dt': '20250101'
                }
            ])
        else:
            try:
                self.mms_df = pd.read_csv(DATA_CONFIG.mms_data_path)
                self.mms_df['msg'] = self.mms_df['msg_nm'] + "\n" + self.mms_df['mms_phrs']
                self.mms_df = (self.mms_df
                              .groupby(["msg_nm", "mms_phrs", "msg"])['offer_dt']
                              .min()
                              .reset_index(name="offer_dt")
                              .reset_index()
                              .astype('str'))
            except Exception as e:
                logger.warning(f"Failed to load real MMS data: {e}. Using mock data instead.")
                self.mms_df = pd.DataFrame([
                    {
                        'msg_nm': 'Sample MMS 1',
                        'mms_phrs': 'Sample MMS content',
                        'msg': 'Sample MMS 1\nSample MMS content',
                        'offer_dt': '20250101'
                    }
                ])
        
        logger.info(f"Loaded {len(self.mms_df)} MMS records")
        
        return self.mms_df
    
    def load_stop_words(self) -> List[str]:
        """
        Load stop words.
        
        Returns:
            List of stop words
        """
        logger.info("Loading stop words")
        
        if self.use_mock_data or not os.path.exists(DATA_CONFIG.stop_words_path):
            logger.info("Using mock stop words")
            self.stop_words = self._create_mock_stop_words()
        else:
            try:
                stop_words_df = pd.read_csv(DATA_CONFIG.stop_words_path)
                self.stop_words = stop_words_df['stop_words'].tolist()
            except Exception as e:
                logger.warning(f"Failed to load real stop words: {e}. Using mock data instead.")
                self.stop_words = self._create_mock_stop_words()
        
        logger.info(f"Loaded {len(self.stop_words)} stop words")
        
        return self.stop_words
    
    def _apply_alias_rules(self, item_nm: str) -> List[str]:
        """
        Apply alias rules to item name.
        
        Args:
            item_nm: Original item name
            
        Returns:
            List of item names including aliases
        """
        item_nm_list = [item_nm]
        
        for rule in self.alias_rules:
            if rule[0] in item_nm:
                item_nm_list.append(item_nm.replace(rule[0], rule[1]))
            if rule[1] in item_nm:
                item_nm_list.append(item_nm.replace(rule[1], rule[0]))
        
        return item_nm_list
    
    def create_entity_vocabulary(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Create entity vocabulary from item data.
        
        Returns:
            List of (entity_name, metadata) tuples
        """
        if self.item_df is None:
            self.load_item_data()
        
        if not self.stop_words:
            self.load_stop_words()
        
        self.entity_vocab = []
        
        for _, row in self.item_df.iterrows():
            if row['item_nm_alias'] not in self.stop_words:
                self.entity_vocab.append((
                    row['item_nm_alias'],
                    {
                        'item_nm': row['item_nm'],
                        'item_id': row['item_id'],
                        'description': row['item_desc'],
                        'domain': row['domain'],
                        'item_nm_alias': row['item_nm_alias']
                    }
                ))
        
        logger.info(f"Created entity vocabulary with {len(self.entity_vocab)} entries")
        
        return self.entity_vocab
    
    def generate_embeddings(self, force_regenerate: bool = False):
        """
        Generate embeddings for items, programs, and organizations.
        
        Args:
            force_regenerate: Whether to force regeneration of embeddings
        """
        if self.embedding_manager is None:
            self.embedding_manager = EmbeddingManager(
                local_model_path=MODEL_CONFIG.local_embedding_model_path,
                loading_mode=MODEL_CONFIG.model_loading_mode
            )
        
        # Generate clue embeddings for programs
        if self.pgm_df is not None and (force_regenerate or self.clue_embeddings is None):
            logger.info("Generating program clue embeddings")
            
            clue_texts = (self.pgm_df[["pgm_nm", "clue_tag"]]
                         .apply(lambda x: preprocess_text(x['pgm_nm'].lower()) + " " + x['clue_tag'].lower(), axis=1)
                         .tolist())
            
            self.clue_embeddings = self.embedding_manager.encode(
                clue_texts,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            logger.info("Generated program clue embeddings")
        
        # Generate item embeddings
        if self.item_df is not None and force_regenerate:
            logger.info("Generating item embeddings")
            
            item_texts = [preprocess_text(x).lower() for x in self.item_df['item_nm_alias'].tolist()]
            
            self.item_embeddings = self.embedding_manager.encode(
                item_texts,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            # Save embeddings
            try:
                self.embedding_manager.save_embeddings(
                    self.item_embeddings,
                    item_texts,
                    DATA_CONFIG.item_embeddings_path
                )
            except Exception as e:
                logger.warning(f"Failed to save item embeddings: {e}")
            
            logger.info("Generated item embeddings")
        
        # Generate organization embeddings
        if self.org_df is not None and force_regenerate:
            logger.info("Generating organization embeddings")
            
            org_texts = (self.org_df[["org_abbr_nm", "bas_addr", "dtl_addr"]]
                        .apply(lambda x: preprocess_text(x['org_abbr_nm'].lower()) + " " + 
                                        x['bas_addr'].lower() + " " + x['dtl_addr'].lower(), axis=1)
                        .tolist())
            
            self.org_embeddings = self.embedding_manager.encode(
                org_texts,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            # Save embeddings
            try:
                self.embedding_manager.save_embeddings(
                    self.org_embeddings,
                    org_texts,
                    DATA_CONFIG.org_all_embeddings_path
                )
            except Exception as e:
                logger.warning(f"Failed to save organization embeddings: {e}")
            
            logger.info("Generated organization embeddings")
    
    def load_embeddings(self):
        """Load pre-computed embeddings from files."""
        if self.embedding_manager is None:
            self.embedding_manager = EmbeddingManager(
                local_model_path=MODEL_CONFIG.local_embedding_model_path,
                loading_mode=MODEL_CONFIG.model_loading_mode
            )
        
        try:
            # Load item embeddings
            if os.path.exists(DATA_CONFIG.item_embeddings_path):
                item_embeddings_np, item_texts = self.embedding_manager.load_embeddings(
                    DATA_CONFIG.item_embeddings_path
                )
                self.item_embeddings = torch.from_numpy(item_embeddings_np)
                logger.info("Loaded item embeddings")
            
            # Load organization embeddings
            if os.path.exists(DATA_CONFIG.org_all_embeddings_path):
                org_embeddings_np, org_texts = self.embedding_manager.load_embeddings(
                    DATA_CONFIG.org_all_embeddings_path
                )
                self.org_embeddings = torch.from_numpy(org_embeddings_np)
                logger.info("Loaded organization embeddings")
                
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
    
    def load_all_data(self, generate_embeddings: bool = True):
        """
        Load all required data.
        
        Args:
            generate_embeddings: Whether to generate embeddings
        """
        logger.info("Loading all data")
        
        # Load basic data
        self.load_item_data()
        self.load_program_data()
        self.load_organization_data()
        self.load_mms_data()
        self.load_stop_words()
        
        # Create entity vocabulary
        self.create_entity_vocabulary()
        
        # Load or generate embeddings
        if generate_embeddings:
            self.load_embeddings()
            self.generate_embeddings(force_regenerate=False)
        
        logger.info("All data loaded successfully")
    
    # Getter methods
    def get_item_data(self) -> pd.DataFrame:
        """Get item data."""
        if self.item_df is None:
            self.load_item_data()
        return self.item_df
    
    def get_program_data(self) -> pd.DataFrame:
        """Get program data."""
        if self.pgm_df is None:
            self.load_program_data()
        return self.pgm_df
    
    def get_organization_data(self) -> pd.DataFrame:
        """Get organization data."""
        if self.org_df is None:
            self.load_organization_data()
        return self.org_df
    
    def get_mms_data(self) -> pd.DataFrame:
        """Get MMS data."""
        if self.mms_df is None:
            self.load_mms_data()
        return self.mms_df
    
    def get_stop_words(self) -> List[str]:
        """Get stop words."""
        if not self.stop_words:
            self.load_stop_words()
        return self.stop_words
    
    def get_entity_vocab(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get entity vocabulary."""
        if not self.entity_vocab:
            self.create_entity_vocabulary()
        return self.entity_vocab
    
    def get_clue_embeddings(self) -> torch.Tensor:
        """Get clue embeddings."""
        if self.clue_embeddings is None:
            self.generate_embeddings()
        return self.clue_embeddings
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embeddings."""
        if self.item_embeddings is None:
            self.load_embeddings()
        return self.item_embeddings
    
    def get_org_embeddings(self) -> torch.Tensor:
        """Get organization embeddings."""
        if self.org_embeddings is None:
            self.load_embeddings()
        return self.org_embeddings 