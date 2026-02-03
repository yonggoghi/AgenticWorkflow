"""
Result Builder Service
=======================

📋 개요
-------
최종 추출 결과를 구성하는 서비스 클래스입니다.
엔티티 매칭, 채널 정보 추출, 프로그램 분류 매핑, offer 객체 생성을 담당합니다.

🔗 의존성
---------
**사용하는 모듈:**
- `services.entity_recognizer`: 엔티티 매칭 및 상품 정보 매핑
- `services.store_matcher`: 대리점/매장 정보 매칭
- `utils.llm_factory`: LLM 모델 생성
- `utils`: 텍스트 정규화 및 성능 로깅

**사용되는 곳:**
- `core.mms_workflow_steps.ResultConstructionStep`: 워크플로우에서 최종 결과 구성

🏗️ 결과 구성 프로세스
--------------------
```mermaid
graph TB
    A[JSON Objects from LLM] --> B[Extract Product Items]
    B --> C{Entity Extraction Mode}
    C -->|logic| D[Logic-based Matching]
    C -->|llm| E[LLM-based Matching]
    D --> F[Similarity DataFrame]
    E --> F
    F --> G[Map Products with Similarity]
    G --> H[Create Offer Object]
    H --> I[Extract Channels]
    I --> J{Channel Type}
    J -->|대리점| K[Match Store Info]
    J -->|기타| L[Keep Original]
    K --> M[Update Offer to org type]
    L --> M
    M --> N[Map Program Classification]
    N --> O[Final Result]
    
    style B fill:#e1f5ff
    style G fill:#ffe1e1
    style H fill:#fff4e1
    style O fill:#e1ffe1
```

📊 스키마 변환
------------

### raw_result (LLM 직접 출력)
```json
{
  "product": [
    {"name": "아이폰 17", "action": "구매"}
  ],
  "channel": [
    {"type": "대리점", "value": "새샘대리점 역곡점"}
  ],
  "pgm": ["5GX 프라임"],
  "purpose": "대리점/매장 방문 유도"
}
```

### ext_result (변환 후 최종 결과)
```json
{
  "product": [
    {
      "item_nm": "아이폰 17",
      "item_id": ["ITEM001"],
      "item_name_in_msg": ["아이폰 17"],
      "expected_action": ["구매"]
    }
  ],
  "channel": [
    {
      "type": "대리점",
      "value": "새샘대리점 역곡점",
      "store_info": [
        {"org_nm": "새샘대리점 역곡점", "org_cd": "ORG001"}
      ]
    }
  ],
  "offer": {
    "type": "org",  // or "product"
    "value": [...]
  },
  "pgm": [
    {"pgm_nm": "5GX 프라임", "pgm_id": "PGM001"}
  ],
  "entity_dag": [],
  "message_id": "MSG001"
}
```

### 스키마 변환 규칙

1. **product 변환**:
   - `name` → `item_nm`
   - DB 매칭으로 `item_id` 추가 (리스트)
   - `name` → `item_name_in_msg` (리스트)
   - `action` → `expected_action` (리스트)

2. **offer 객체 생성**:
   - 기본: `type='product'`, `value=product 리스트`
   - 대리점 감지 시: `type='org'`, `value=매장 정보`

3. **channel 보강**:
   - 대리점 타입: `store_info` 추가 (StoreMatch 결과)
   - 기타 타입: `store_info=[]`

4. **pgm 매핑**:
   - LLM 추출 프로그램명 → DB 매칭
   - `pgm_nm`, `pgm_id` 추가

🏗️ 주요 컴포넌트
----------------
- **ResultBuilder**: 결과 구성 서비스 클래스
  - `build_extraction_result()`: 전체 결과 구성 파이프라인
  - `_map_program_classification()`: 프로그램 분류 매핑
  - `_extract_channels()`: 채널 정보 추출 및 offer 업데이트

💡 사용 예시
-----------
```python
from services.result_builder import ResultBuilder
from utils.llm_factory import LLMFactory

# 초기화
llm_factory = LLMFactory()
builder = ResultBuilder(
    entity_recognizer=recognizer,
    store_matcher=matcher,
    alias_pdf_raw=alias_df,
    stop_item_names=['광고', '이벤트'],
    num_cand_pgms=10,
    entity_extraction_mode='llm',
    llm_factory=llm_factory,
    llm_model='ax',
    entity_extraction_context_mode='dag'
)

# 최종 결과 구성
final_result = builder.build_extraction_result(
    json_objects=llm_response,
    msg="아이폰 17 구매 시 캐시백 제공",
    pgm_info=program_info,
    entities_from_kiwi=['아이폰', '캐시백'],
    message_id='MSG001'
)

print(f"추출된 상품 수: {len(final_result['product'])}")
print(f"Offer 타입: {final_result['offer']['type']}")
```

📝 참고사항
----------
- entity_extraction_mode='logic': Fuzzy + Sequence 유사도 사용
- entity_extraction_mode='llm': LLM 기반 2단계 추출
- 대리점 감지 시 offer 타입이 자동으로 'org'로 변경
- 매칭 실패 시 LLM 원본 결과 사용 (item_id='#')
- 모든 리스트 필드는 중복 제거 적용

"""

import logging
import pandas as pd
import re
import os
import torch
from typing import List, Dict, Any, Tuple, Optional
from langchain_openai import ChatOpenAI
from utils import (
    replace_special_chars_with_space,
    log_performance
)
from config.settings import MODEL_CONFIG, API_CONFIG, PROCESSING_CONFIG

logger = logging.getLogger(__name__)

class ResultBuilder:
    """
    Service for building the final extraction result.
    Handles entity matching, channel extraction, and result assembly.
    """

    def __init__(self, entity_recognizer, store_matcher, alias_pdf_raw: pd.DataFrame, 
                 stop_item_names: List[str], num_cand_pgms: int, entity_extraction_mode: str,
                 llm_factory=None,  # LLMFactory 인스턴스 (기존 callable 대체)
                 llm_model: str = 'ax', 
                 entity_extraction_context_mode: str = 'dag'):
        self.entity_recognizer = entity_recognizer
        self.store_matcher = store_matcher
        self.alias_pdf_raw = alias_pdf_raw
        self.stop_item_names = stop_item_names
        self.num_cand_pgms = num_cand_pgms
        self.entity_extraction_mode = entity_extraction_mode
        self.llm_factory = llm_factory  # LLMFactory 인스턴스 저장
        self.llm_model = llm_model
        self.entity_extraction_context_mode = entity_extraction_context_mode

    def build_extraction_result(self, json_objects: Dict, msg: str, pgm_info: Dict, entities_from_kiwi: List[str], message_id: str = '#') -> Dict[str, Any]:
        """최종 결과 구성"""
        try:
            logger.info("=" * 80)
            logger.info("🔍 [PRODUCT DEBUG] build_extraction_result 시작")
            logger.info("=" * 80)
            
            final_result = json_objects.copy()
            
            # offer_object 초기화
            offer_object = {}
            
            # 상품 정보에서 엔티티 추출
            logger.info("📋 [STEP 1] product_items 추출")
            product_items = json_objects.get('product', [])
            logger.info(f"   - 원본 product 타입: {type(product_items)}")
            logger.info(f"   - 원본 product 내용: {product_items}")
            
            if isinstance(product_items, dict):
                logger.info("   - product가 dict 타입 → 'items' 키로 접근")
                product_items = product_items.get('items', [])
                logger.info(f"   - items 추출 후: {product_items}")
            
            logger.info(f"   ✅ 최종 product_items 개수: {len(product_items)}개")
            logger.info(f"   ✅ 최종 product_items 내용: {product_items}")

            primary_llm_extracted_entities = [x.get('name', '') for x in product_items]
            logger.info(f"📋 [STEP 2] LLM 추출 엔티티: {primary_llm_extracted_entities}")
            logger.info(f"📋 [STEP 2] Kiwi 엔티티: {entities_from_kiwi}")
            logger.info(f"📋 [STEP 2] entity_extraction_mode: {self.entity_extraction_mode}")

            # 엔티티 매칭 모드에 따른 처리
            if self.entity_extraction_mode == 'logic':
                logger.info("🔍 [STEP 3] 로직 기반 엔티티 매칭 시작")
                # 로직 기반: 퍼지 + 시퀀스 유사도
                cand_entities = list(set(entities_from_kiwi+[item.get('name', '') for item in product_items if item.get('name')]))
                logger.info(f"   - cand_entities: {cand_entities}")
                similarities_fuzzy = self.entity_recognizer.extract_entities_with_fuzzy_matching(cand_entities)
                logger.info(f"   ✅ similarities_fuzzy 결과 크기: {similarities_fuzzy.shape if not similarities_fuzzy.empty else '비어있음'}")
            else:
                logger.info("🔍 [STEP 3] LLM 기반 엔티티 매칭 시작")
                logger.info(f"   - 사용할 LLM 모델: {self.llm_model}")
                logger.info(f"   - llm_factory 타입: {type(self.llm_factory)}")
                
                # LLMFactory를 사용하여 모델 생성
                if self.llm_factory:
                    default_llm_models = self.llm_factory.create_models([self.llm_model])
                    logger.info(f"   - 초기화된 LLM 모델 수: {len(default_llm_models)}개")
                else:
                    logger.warning("⚠️ llm_factory가 설정되지 않았습니다. 빈 리스트를 사용합니다.")
                    default_llm_models = []
                logger.info(f"   - 초기화된 LLM 모델 수: {len(default_llm_models)}개")
                llm_result = self.entity_recognizer.extract_entities_with_llm(
                    msg,
                    llm_models=default_llm_models,
                    rank_limit=100,
                    external_cand_entities=list(set(entities_from_kiwi+primary_llm_extracted_entities)),
                    context_mode=self.entity_extraction_context_mode
                )

                # ONT 모드일 경우 dict 반환, 그 외는 DataFrame 반환
                if isinstance(llm_result, dict):
                    similarities_fuzzy = llm_result.get('similarities_df', pd.DataFrame())
                    logger.info(f"   ✅ ONT 모드: similarities_df 추출 완료")
                else:
                    similarities_fuzzy = llm_result

                logger.info(f"   ✅ similarities_fuzzy 결과 크기: {similarities_fuzzy.shape if not similarities_fuzzy.empty else '비어있음'}")
            
            if not similarities_fuzzy.empty:
                logger.info(f"   📊 similarities_fuzzy 샘플 (처음 3개):")
                logger.info(f"{similarities_fuzzy.head(3).to_dict('records')}")
            else:
                logger.warning("   ⚠️ similarities_fuzzy가 비어있습니다!")

            if not similarities_fuzzy.empty:
                logger.info("🔍 [STEP 4] alias_pdf_raw와 merge 시작")
                logger.info(f"   - alias_pdf_raw 크기: {self.alias_pdf_raw.shape}")
                merged_df = similarities_fuzzy.merge(
                    self.alias_pdf_raw[['alias_1','type']].drop_duplicates(), 
                    left_on='item_name_in_msg', 
                    right_on='alias_1', 
                    how='left'
                )
                logger.info(f"   ✅ merged_df 크기: {merged_df.shape if not merged_df.empty else '비어있음'}")
                if not merged_df.empty:
                    logger.info(f"   📊 merged_df 샘플 (처음 3개):")
                    logger.info(f"{merged_df.head(3).to_dict('records')}")

                logger.info("🔍 [STEP 5] filtered_df 생성 (expansion 타입 필터링)")
                filtered_df = merged_df[merged_df.apply(
                    lambda x: (
                        replace_special_chars_with_space(x['item_nm_alias']) in replace_special_chars_with_space(x['item_name_in_msg']) or 
                        replace_special_chars_with_space(x['item_name_in_msg']) in replace_special_chars_with_space(x['item_nm_alias'])
                    ) if x['type'] != 'expansion' else True, 
                    axis=1
                )]
                logger.info(f"   ✅ filtered_df 크기: {filtered_df.shape if not filtered_df.empty else '비어있음'}")
                if not filtered_df.empty:
                    logger.info(f"   📊 filtered_df 샘플 (처음 3개):")
                    logger.info(f"{filtered_df.head(3).to_dict('records')}")

                # similarities_fuzzy = filtered_df[similarities_fuzzy.columns]

            # 상품 정보 매핑
            logger.info("🔍 [STEP 6] 상품 정보 매핑 시작")
            logger.info(f"   - similarities_fuzzy.empty: {similarities_fuzzy.empty}")
            
            if not similarities_fuzzy.empty:
                logger.info("   ✅ similarities_fuzzy가 비어있지 않음 → map_products_with_similarity 호출")
                final_result['product'] = self.entity_recognizer.map_products_to_entities(similarities_fuzzy, json_objects)
                logger.info(f"   ✅ 최종 product 개수: {len(final_result['product'])}개")
                logger.info(f"   ✅ 최종 product 내용: {final_result['product']}")
            else:
                logger.warning("   ⚠️ similarities_fuzzy가 비어있음 → LLM 결과 그대로 사용 (else 브랜치)")
                logger.info(f"   - product_items 개수: {len(product_items)}개")
                logger.info(f"   - stop_item_names 개수: {len(self.stop_item_names)}개")
                
                # 유사도 결과가 없으면 LLM 결과 그대로 사용 (새 스키마 + expected_action 리스트)
                filtered_product_items = [
                    d for d in product_items 
                    if d.get('name') and d['name'] not in self.stop_item_names
                ]
                logger.info(f"   - 필터링 후 product_items 개수: {len(filtered_product_items)}개")
                logger.info(f"   - 필터링 후 product_items: {filtered_product_items}")
                
                final_result['product'] = [
                    {
                        'item_nm': d.get('name', ''), 
                        'item_id': ['#'],
                        'item_name_in_msg': [d.get('name', '')],
                        'expected_action': [d.get('action', '기타')]
                    } 
                    for d in filtered_product_items
                ]
                logger.info(f"   ✅ 최종 product 개수: {len(final_result['product'])}개")
                logger.info(f"   ✅ 최종 product 내용: {final_result['product']}")

            # offer_object에 product 타입으로 설정
            offer_object['type'] = 'product'
            offer_object['value'] = final_result['product']
            logger.info(f"🏷️  [STEP 7] offer_object 초기화: type=product, value 개수={len(offer_object['value'])}개")

            # 프로그램 분류 정보 매핑
            final_result['pgm'] = self._map_programs_to_result(json_objects, pgm_info)
            
            # 채널 정보 처리 (offer_object도 함께 전달 및 반환)
            logger.info("🔍 [STEP 8] 채널 정보 처리 및 offer_object 업데이트")
            final_result['channel'], offer_object = self._extract_and_enrich_channels(json_objects, msg, offer_object)
            logger.info(f"   ✅ 최종 channel 개수: {len(final_result['channel'])}개")
            logger.info(f"   ✅ 최종 offer_object type: {offer_object.get('type', 'N/A')}")
            logger.info(f"   ✅ 최종 offer_object value 개수: {len(offer_object.get('value', []))}개")
            
            # offer 필드 추가
            final_result['offer'] = offer_object
            logger.info(f"✅ [STEP 9] final_result에 offer 필드 추가 완료")
            
            # entity_dag 초기화 (빈 배열)
            final_result['entity_dag'] = []
            
            logger.info("=" * 80)
            logger.info("✅ [PRODUCT DEBUG] build_extraction_result 완료")
            logger.info(f"   최종 final_result['product'] 개수: {len(final_result.get('product', []))}개")
            logger.info("=" * 80)

            # message_id 추가
            final_result['message_id'] = message_id
            
            return final_result
            
        except Exception as e:
            logger.error(f"최종 결과 구성 실패: {e}")
            return json_objects

    def _map_programs_to_result(self, json_objects: Dict, pgm_info: Dict) -> List[Dict]:
        """프로그램 분류 정보 매핑"""
        try:
            if (self.num_cand_pgms > 0 and 
                'pgm' in json_objects and 
                isinstance(json_objects['pgm'], list) and
                not pgm_info.get('pgm_pdf_tmp', pd.DataFrame()).empty):
                
                pgm_json = pgm_info['pgm_pdf_tmp'][
                    pgm_info['pgm_pdf_tmp']['pgm_nm'].apply(
                        lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm'])
                    )
                ][['pgm_nm', 'pgm_id']].to_dict('records')
                
                return pgm_json
            
            return []
            
        except Exception as e:
            logger.error(f"프로그램 분류 매핑 실패: {e}")
            return []

    def _extract_and_enrich_channels(self, json_objects: Dict, msg: str, offer_object: Dict) -> Tuple[List[Dict], Dict]:
        """채널 정보 추출 및 매칭 (offer_object도 함께 반환)"""
        try:
            channel_tag = []
            channel_items = json_objects.get('channel', [])
            if isinstance(channel_items, dict):
                channel_items = channel_items.get('items', [])

            for d in channel_items:
                if d.get('type') == '대리점' and d.get('value'):
                    # 대리점명으로 조직 정보 검색
                    store_info = self.store_matcher.match_store(d['value'])
                    d['store_info'] = store_info
                    
                    # offer_object를 org 타입으로 변경
                    if store_info:
                        offer_object['type'] = 'org'
                        org_tmp = [
                            {
                                'item_nm': o['org_nm'], 
                                'item_id': o['org_cd'], 
                                'item_name_in_msg': d['value'], 
                                'expected_action': ['방문']
                            } 
                            for o in store_info
                        ]
                        offer_object['value'] = org_tmp
                    else:
                        if "대리점/매장 방문 유도" in json_objects['purpose']:
                            offer_object['type'] = 'org'
                            org_tmp = [{'item_nm':d['value'], 'item_id':'#', 'item_name_in_msg':d['value'], 'expected_action':['방문']}]
                            offer_object['value'] = org_tmp
                else:
                    d['store_info'] = []
                channel_tag.append(d)

            return channel_tag, offer_object
            
        except Exception as e:
            logger.error(f"채널 정보 추출 실패: {e}")
            return [], offer_object


