import pandas as pd
import numpy as np
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union, Dict, Any
import ast
from enum import Enum

# Set pandas display options
pd.set_option('display.max_colwidth', 500)

class ExtractionApproach(Enum):
    """Enumeration for different extraction approaches"""
    ENTITY_ASSISTED = "entity_assisted"
    LLM_ONLY = "llm_only" 
    LLM_COT = "llm_cot"

class MMSExtractor:
    """Main class for MMS message extraction with multiple approaches"""
    
    def __init__(self):
        # Initialize alias_rule_set first
        self.alias_rule_set = []
        self.setup_clients()
        self.load_data()
        self.setup_entity_matcher()
        self.setup_models()
        
    def setup_clients(self):
        """Setup API clients"""
        from openai import OpenAI
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        
        # API configurations
        self.llm_api_key = config.CUSTOM_API_KEY"https://api.platform.a15t.com/v1"
        
        self.client = OpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_api_url
        )
        
        # Import configuration
        from config import config
        
        # Setup different LLM clients
        self.llm_cld35 = self._create_chat_anthropic_skt()
        self.llm_cld37 = ChatAnthropic(
            api_key=config.ANTHROPIC_API_KEY,
            model="claude-3-7-sonnet-20250219",
            max_tokens=3000
        )
        
    def _create_chat_anthropic_skt(self, model="skt/claude-3-5-sonnet-20241022", max_tokens=100):
        """Create ChatAnthropic SKT client"""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            temperature=0,
            openai_api_key=self.llm_api_key,
            openai_api_base=self.llm_api_url,
            model=model,
            max_tokens=max_tokens
        )
        
    def load_data(self):
        """Load all required data files"""
        # Load MMS data
        self.mms_pdf = pd.read_csv("./data/mms_data_250408.csv")
        self._process_mms_data()
        
        # Load item data
        self.item_pdf_raw = pd.read_csv("./data/item_info_all_250527.csv")
        
        # Load stop words
        self.stop_item_names = pd.read_csv("./data/stop_words.csv")['stop_words'].to_list()
        self.stop_item_names = list(set(self.stop_item_names + [x.lower() for x in self.stop_item_names]))
        
        # Load alias rules
        alias_df = pd.read_csv("./data/alias_rules_ke.csv")
        self.alias_rule_set = list(zip(alias_df['korean'], alias_df['english']))
        
        # Process item data after alias rules are loaded
        self._process_item_data()
        
        # Load program data
        self.pgm_pdf = pd.read_csv("./data/pgm_tag_ext_250516.csv")
        
    def _process_mms_data(self):
        """Process MMS data"""
        self.mms_pdf['msg'] = self.mms_pdf['msg_nm'] + "\n" + self.mms_pdf['mms_phrs']
        self.mms_pdf = self.mms_pdf.groupby(["msg_nm", "mms_phrs", "msg"])['offer_dt'].min().reset_index(name="offer_dt")
        self.mms_pdf = self.mms_pdf.reset_index().astype('str')
        
    def _process_item_data(self):
        """Process item data and create entity list"""
        self.item_pdf_all = self.item_pdf_raw.drop_duplicates(['item_nm', 'item_id'])[
            ['item_nm', 'item_id', 'item_desc', 'domain', 'start_dt', 'end_dt', 'rank']
        ].copy()
        
        # Apply alias rules
        self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm'].apply(self._apply_alias_rule)
        self.item_pdf_all = self.item_pdf_all.explode('item_nm_alias')
        
        # Add user-defined entities
        user_defined_entity = ['AIA Vitality', '부스트 파크 건대입구', 'Boost Park 건대입구']
        item_pdf_ext = pd.DataFrame([
            {
                'item_nm': e, 'item_id': e, 'item_desc': e, 'domain': 'user_defined',
                'start_dt': 20250101, 'end_dt': 99991231, 'rank': 1, 'item_nm_alias': e
            } for e in user_defined_entity
        ])
        self.item_pdf_all = pd.concat([self.item_pdf_all, item_pdf_ext])
        
        # Create entity list for fuzzy matching
        self.entity_list_for_fuzzy = []
        for row in self.item_pdf_all.to_dict('records'):
            self.entity_list_for_fuzzy.append((
                row['item_nm'], 
                {
                    'item_id': row['item_id'],
                    'description': row['item_desc'],
                    'domain': row['domain'],
                    'start_dt': row['start_dt'],
                    'end_dt': row['end_dt'],
                    'rank': 1,
                    'item_nm_alias': row['item_nm_alias']
                }
            ))
            
    def _apply_alias_rule(self, item_nm):
        """Apply alias rules to item names"""
        item_nm_list = [item_nm]
        for korean, english in self.alias_rule_set:
            if korean in item_nm:
                item_nm_list.append(item_nm.replace(korean, english))
            if english in item_nm:
                item_nm_list.append(item_nm.replace(english, korean))
        return item_nm_list
        
    def setup_entity_matcher(self):
        """Setup Korean entity matcher and related components"""
        from kiwipiepy import Kiwi
        
        self.kiwi = Kiwi()
        self.kiwi_raw = Kiwi()
        self.kiwi_raw.space_tolerance = 2
        
        # Add user words to Kiwi
        entity_list_for_kiwi = list(self.item_pdf_all['item_nm_alias'].unique())
        for w in entity_list_for_kiwi:
            self.kiwi.add_user_word(w, "NNP")
        for w in self.stop_item_names:
            self.kiwi.add_user_word(w, "NNG")
            
        self.tags_to_exclude = ['W_SERIAL', 'W_URL', 'JKO', 'SSO', 'SSC', 'SW', 'SF', 'SP', 'SS', 'SE', 'SO', 'SB', 'SH']
        
    def setup_models(self):
        """Setup sentence transformer model"""
        from sentence_transformers import SentenceTransformer
        import torch
        
        self.model = SentenceTransformer('jhgan/ko-sbert-nli')
        
        # Precompute embeddings for program classification
        def preprocess_text(text):
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
        self.clue_embeddings = self.model.encode(
            self.pgm_pdf[["pgm_nm", "clue_tag"]].apply(
                lambda x: preprocess_text(x['pgm_nm'].lower()) + " " + x['clue_tag'].lower(), 
                axis=1
            ).tolist(), 
            convert_to_tensor=True
        )
        
    def extract_entities_with_kiwi(self, text):
        """Extract entities using Kiwi tokenizer"""
        result_msg = self.kiwi.tokenize(text, normalize_coda=True, z_coda=False, split_complex=False)
        entities_from_kiwi = []
        
        for token in result_msg:
            if (token.tag == 'NNP' and 
                token.form not in self.stop_item_names + ['-'] and 
                len(token.form) >= 2 and 
                token.form.lower() not in self.stop_item_names):
                entities_from_kiwi.append(token.form)
                
        return self._filter_specific_terms(entities_from_kiwi)
        
    def _filter_specific_terms(self, strings: List[str]) -> List[str]:
        """Filter overlapping terms, keeping longer ones"""
        unique_strings = list(set(strings))
        unique_strings.sort(key=len, reverse=True)
        
        filtered = []
        for s in unique_strings:
            if not any(s in other for other in filtered):
                filtered.append(s)
        return filtered
        
    def find_program_candidates(self, message, num_candidates=5):
        """Find program classification candidates using semantic similarity"""
        import torch
        
        mms_embedding = self.model.encode([message.lower()], convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(
            mms_embedding, self.clue_embeddings, dim=1
        ).cpu().numpy()
        
        pgm_pdf_tmp = self.pgm_pdf.copy()
        pgm_pdf_tmp['sim'] = similarities
        pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)
        
        return pgm_pdf_tmp.iloc[:num_candidates]
        
    def entity_assisted_extraction(self, message):
        """Entity-assisted extraction approach"""
        print("Running Entity-Assisted Extraction...")
        
        # Extract entities and find products
        entities_from_kiwi = self.extract_entities_with_kiwi(message)
        product_df = self._process_entities_for_products(message, entities_from_kiwi)
        
        # Setup schema with product elements
        product_element = product_df.to_dict(orient='records') if not product_df.empty else []
        
        schema = {
            "title": {"type": "string", "description": "광고 제목"},
            "purpose": {"type": "array", "description": "광고의 주요 목적"},
            "product": product_element,
            "channel": {"type": "array", "items": {"type": "object", "properties": {
                "type": {"type": "string"}, "value": {"type": "string"}, 
                "action": {"type": "string"}, "store_code": {"type": "string"}
            }}},
            "pgm": {"type": "array", "description": "프로그램 분류"}
        }
        
        return self._call_llm_extraction(message, schema, "entity_assisted")
        
    def llm_only_extraction(self, message):
        """LLM-only extraction approach"""
        print("Running LLM-Only Extraction...")
        
        schema = {
            "title": {"type": "string", "description": "광고 제목"},
            "purpose": {"type": "array", "description": "광고의 주요 목적"},
            "product": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "action": {"type": "string"}
            }}},
            "channel": {"type": "array", "items": {"type": "object", "properties": {
                "type": {"type": "string"}, "value": {"type": "string"}, 
                "action": {"type": "string"}, "store_code": {"type": "string"}
            }}},
            "pgm": {"type": "array", "description": "프로그램 분류"}
        }
        
        return self._call_llm_extraction(message, schema, "llm_only")
        
    def llm_cot_extraction(self, message):
        """LLM Chain-of-Thought extraction approach"""
        print("Running LLM Chain-of-Thought Extraction...")
        
        schema = {
            "reasoning": {"type": "object", "description": "단계별 분석 과정"},
            "title": {"type": "string", "description": "광고 제목"},
            "purpose": {"type": "array", "description": "광고의 주요 목적"},
            "product": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "action": {"type": "string"}
            }}},
            "channel": {"type": "array", "items": {"type": "object", "properties": {
                "type": {"type": "string"}, "value": {"type": "string"}, 
                "action": {"type": "string"}, "store_code": {"type": "string"}
            }}},
            "pgm": {"type": "array", "description": "프로그램 분류"}
        }
        
        return self._call_llm_extraction(message, schema, "llm_cot")
        
    def _call_llm_extraction(self, message, schema, approach_type):
        """Call LLM for extraction with appropriate prompt"""
        # Get program candidates
        pgm_candidates = self.find_program_candidates(message)
        pgm_info = "\n\t".join(
            pgm_candidates[['pgm_nm', 'clue_tag']].apply(
                lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm']) + " : " + x['clue_tag'], 
                axis=1
            ).to_list()
        )
        
        # Create prompt based on approach
        if approach_type == "llm_cot":
            prompt = self._create_cot_prompt(message, schema, pgm_info)
            model = "skt/a.x-3-lg"
        else:
            prompt = self._create_standard_prompt(message, schema, pgm_info, approach_type)
            model = "skt/claude-3-5-sonnet-20241022" if approach_type == "entity_assisted" else "skt/a.x-3-lg"
            
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            result_json_text = response.choices[0].message.content
            json_objects = self._extract_json_objects(result_json_text)[0]
            
            # Post-process results
            return self._post_process_results(json_objects, pgm_candidates)
            
        except Exception as e:
            print(f"Error with API call: {e}")
            return None
            
    def _create_standard_prompt(self, message, schema, pgm_info, approach_type):
        """Create standard extraction prompt"""
        context = f"\n\n### 광고 분류 기준 정보 ###\n\t{pgm_info}" if pgm_info else ""
        
        return f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. 아래 schema에 따라 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요:

### 분석 대상 광고 메세지 ###
{message}

### 결과 Schema ###
{json.dumps(schema, indent=2, ensure_ascii=False)}

### 분석 지침 ###
- 재현율이 높도록 모든 상품을 선택
- 광고 분류 기준 정보는 pgm_nm : clue_tag 로 구성
- JSON 형식으로만 응답 (추가 설명 없이)

{context}
"""
        
    def _create_cot_prompt(self, message, schema, pgm_info):
        """Create Chain-of-Thought prompt"""
        context = f"\n\n### 광고 분류 기준 정보 ###\n\t{pgm_info}" if pgm_info else ""
        
        return f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. **단계별 사고 과정(Chain of Thought)**을 통해 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요.

## 분석 단계 (Chain of Thought)

### STEP 1: 광고 목적(Purpose) 분석
먼저 광고 메시지 전체를 읽고 다음 질문들에 답하여 광고의 주요 목적을 파악하세요:
- 이 광고가 고객에게 무엇을 하라고 요구하는가?
- 어떤 행동을 유도하려고 하는가?
- 어떤 혜택이나 정보를 제공하고 있는가?

### STEP 2: 상품(Product) 식별
- 광고 메시지에서 언급된 모든 상품/서비스 추출
- 각 상품별 고객 행동(Action) 결정

### STEP 3: 채널(Channel) 추출
- URL, 전화번호, 앱, 대리점 정보 추출
- 각 채널의 목적과 혜택 파악

### STEP 4: 프로그램 분류(PGM) 결정
- 광고 분류 기준 정보와 메시지 내용 매칭
- 적합도 순서대로 2개 선택

### 분석 대상 광고 메세지 ###
{message}

### 결과 Schema ###
{json.dumps(schema, indent=2, ensure_ascii=False)}

### 분석 지침 ###
- reasoning 섹션은 분석 과정 설명용 (최종 JSON에는 포함하지 않음)
- 순수한 JSON 형식으로만 응답
- 추가 텍스트나 설명 없이 JSON만 제공

{context}
"""
        
    def _process_entities_for_products(self, message, entities_from_kiwi):
        """Process entities to create product dataframe with matched items from database"""
        if not entities_from_kiwi:
            return pd.DataFrame()
            
        # Create a list to store matched products
        matched_products = []
        
        # Process each entity
        for entity in entities_from_kiwi:
            # Find exact matches in item database
            exact_matches = self.item_pdf_all[
                (self.item_pdf_all['item_nm_alias'] == entity) |
                (self.item_pdf_all['item_nm'] == entity)
            ]
            
            # If no exact matches, try fuzzy matching
            if exact_matches.empty:
                # Find similar items using entity list
                similar_items = []
                for item_name, item_info in self.entity_list_for_fuzzy:
                    if entity.lower() in item_name.lower() or item_name.lower() in entity.lower():
                        similar_items.append((item_name, item_info))
                
                # Add matched items to products list
                for item_name, item_info in similar_items:
                    matched_products.append({
                        'name': item_name,
                        'item_id': item_info['item_id'],
                        'description': item_info['description'],
                        'domain': item_info['domain'],
                        'action': self._determine_action(message, item_name)
                    })
            else:
                # Add exact matches to products list
                for _, row in exact_matches.iterrows():
                    matched_products.append({
                        'name': row['item_nm'],
                        'item_id': row['item_id'],
                        'description': row['item_desc'],
                        'domain': row['domain'],
                        'action': self._determine_action(message, row['item_nm'])
                    })
        
        # Convert to DataFrame and remove duplicates
        if matched_products:
            product_df = pd.DataFrame(matched_products)
            product_df = product_df.drop_duplicates(subset=['item_id'])
            return product_df
        return pd.DataFrame()
        
    def _determine_action(self, message: str, item_name: str) -> str:
        """Determine the action associated with an item based on message context"""
        # Common action patterns
        action_patterns = {
            '가입': ['가입', '신청', '가입하세요', '신청하세요', '가입하기', '신청하기'],
            '구매': ['구매', '구입', '구매하세요', '구입하세요', '구매하기', '구입하기'],
            '이용': ['이용', '사용', '이용하세요', '사용하세요', '이용하기', '사용하기'],
            '다운로드': ['다운로드', '설치', '다운로드하세요', '설치하세요', '다운로드하기', '설치하기'],
            '방문': ['방문', '찾아오세요', '방문하세요', '찾아오기', '방문하기'],
            '쿠폰': ['쿠폰', '할인', '쿠폰받기', '할인받기', '쿠폰받으세요', '할인받으세요']
        }
        
        # Find the item name in the message
        item_idx = message.lower().find(item_name.lower())
        if item_idx == -1:
            return "정보제공"  # Default action if item not found in message
            
        # Look for action patterns around the item name
        context_window = 50  # Characters to look before and after item name
        start_idx = max(0, item_idx - context_window)
        end_idx = min(len(message), item_idx + len(item_name) + context_window)
        context = message[start_idx:end_idx].lower()
        
        # Check each action pattern
        for action, patterns in action_patterns.items():
            if any(pattern in context for pattern in patterns):
                return action
                
        return "정보제공"  # Default action if no specific action found
        
    def _extract_json_objects(self, text):
        """Extract JSON objects from text"""
        pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
        result = []
        
        for match in re.finditer(pattern, text):
            potential_json = match.group(0)
            try:
                json_obj = json.loads(potential_json)
                result.append(json_obj)
            except json.JSONDecodeError:
                try:
                    json_obj = ast.literal_eval(potential_json)
                    result.append(json_obj)
                except:
                    pass
        return result
        
    def _post_process_results(self, json_objects, pgm_candidates):
        """Post-process extraction results"""
        # Add program classification
        if 'pgm' in json_objects and json_objects['pgm']:
            pgm_json = self.pgm_pdf[
                self.pgm_pdf['pgm_nm'].apply(
                    lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm'])
                )
            ][['pgm_nm', 'pgm_id']].to_dict('records')
            json_objects['pgm'] = pgm_json
            
        return json_objects
        
    def extract(self, message, approach: ExtractionApproach):
        """Main extraction method"""
        if approach == ExtractionApproach.ENTITY_ASSISTED:
            return self.entity_assisted_extraction(message)
        elif approach == ExtractionApproach.LLM_ONLY:
            return self.llm_only_extraction(message)
        elif approach == ExtractionApproach.LLM_COT:
            return self.llm_cot_extraction(message)
        else:
            raise ValueError(f"Unknown approach: {approach}")

def main():
    """Main function to run the extraction"""
    
    # Sample messages for testing
    msg_text_list = [
        """
        광고 제목:[SK텔레콤] 2월 0 day 혜택 안내
        광고 내용:(광고)[SKT] 2월 0 day 혜택 안내__[2월 10일(토) 혜택]_만 13~34세 고객이라면_베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!_(선착순 3만 명 증정)_▶ 자세히 보기: http://t-mms.kr/t.do?m=#61&s=24589&a=&u=https://bit.ly/3SfBjjc__■ 에이닷 X T 멤버십 시크릿코드 이벤트_에이닷 T 멤버십 쿠폰함에 '에이닷이빵쏜닷'을 입력해보세요!_뚜레쥬르 데일리우유식빵 무료 쿠폰을 드립니다._▶ 시크릿코드 입력하러 가기: https://bit.ly/3HCUhLM__■ 문의: SKT 고객센터(1558, 무료)_무료 수신거부 1504
        """,
        """
        '[T 우주] 넷플릭스와 웨이브를 월 9,900원에! 
        (광고)[SKT] 넷플릭스+웨이브 월 9,900원, 이게 되네! __#04 고객님,_넷플릭스와 웨이브 둘 다 보고 싶었지만, 가격 때문에 망설이셨다면 지금이 바로 기회! __오직 T 우주에서만, _2개월 동안 월 9,900원에 넷플릭스와 웨이브를 모두 즐기실 수 있습니다.__8월 31일까지만 드리는 혜택이니, 지금 바로 가입해 보세요! __■ 우주패스 Netflix 런칭 프로모션 _- 기간 : 2024년 8월 31일(토)까지_- 혜택 : 우주패스 Netflix(광고형 스탠다드)를 2개월 동안 월 9,900원에 이용 가능한 쿠폰 제공_▶ 프로모션 자세히 보기: http://t-mms.kr/jAs/#74__■ 우주패스 Netflix(월 12,000원)  _- 기본 혜택 : Netflix 광고형 스탠다드 멤버십_- 추가 혜택 : Wavve 콘텐츠 팩 _* 추가 요금을 내시면 Netflix 스탠다드와 프리미엄 멤버십 상품으로 가입 가능합니다.  __■ 유의 사항_-  프로모션 쿠폰은 1인당 1회 다운로드 가능합니다. _-  쿠폰 할인 기간이 끝나면 정상 이용금액으로 자동 결제 됩니다. __■ 문의: T 우주 고객센터 (1505, 무료)__나만의 구독 유니버스, T 우주 __무료 수신거부 1504'
        """
    ]
    
    # Configuration
    MESSAGE_IDX = 0  # Choose which message to process (0 or 1)
    APPROACH = ExtractionApproach.ENTITY_ASSISTED  # Choose approach: ENTITY_ASSISTED, LLM_ONLY, or LLM_COT
    
    print(f"Selected approach: {APPROACH.value}")
    print("=" * 50)
    
    # Initialize extractor
    print("Initializing MMS Extractor...")
    try:
        extractor = MMSExtractor()
        print("✓ Extractor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize extractor: {e}")
        return
    
    # Get message
    mms_msg = msg_text_list[MESSAGE_IDX]
    print(f"\nProcessing message {MESSAGE_IDX + 1}:")
    print("-" * 30)
    print(mms_msg[:200] + "..." if len(mms_msg) > 200 else mms_msg)
    print("-" * 30)
    
    # Run extraction
    print(f"\nRunning {APPROACH.value} extraction...")
    start_time = time.time()
    
    try:
        result = extractor.extract(mms_msg, APPROACH)
        end_time = time.time()
        
        print(f"✓ Extraction completed in {end_time - start_time:.2f} seconds")
        print("\nResults:")
        print("=" * 50)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()