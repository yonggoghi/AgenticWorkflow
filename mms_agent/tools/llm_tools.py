"""
LLM-based tools for MMS Agent
Uses simplified prompts and existing LLM configuration
"""

import json
import logging
from typing import Dict, Any, List
from langchain.tools import tool

from ..core.llm_client import get_llm
from ..core import ExtractorBase

logger = logging.getLogger(__name__)

# Singleton instances
_llm = get_llm()
_extractor = ExtractorBase()


# Simple prompts for LLM tools
ENTITY_EXTRACTION_PROMPT = """다음 MMS 광고 메시지에서 상품/서비스명을 추출하세요.

메시지:
{message}

요구사항:
1. 원문에 명시된 상품/서비스명만 추출
2. 한국어는 그대로, 영어는 그대로 유지
3. 콤마로 구분하여 나열

출력 형식:
상품1, 상품2, 상품3
"""

MAIN_INFO_PROMPT = """다음 MMS 광고 메시지에서 정보를 추출하세요.

메시지:
{message}

{context}

다음 JSON 형식으로 출력하세요:
{{
  "title": "광고 제목 (50자 이내)",
  "purpose": ["목적1", "목적2"],
  "product": [
    {{"name": "상품명", "action": "가입|구매|사용|방문|참여"}}
  ],
  "channel": [
    {{"type": "URL|전화번호|앱|대리점", "value": "값", "action": "가입|문의|방문"}}
  ],
  "sales_script": "콜센터용 짧은 스크립트"
}}

목적 카테고리: 상품 가입 유도, 대리점/매장 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 수신 거부 안내, 기타 정보 제공
"""

DAG_EXTRACTION_PROMPT = """다음 광고 메시지에서 사용자 행동 흐름을 DAG로 추출하세요.

메시지:
{message}

출력 형식:
(엔티티:행동) -[관계]-> (엔티티:행동)

예시:
(대리점:방문) -[방문하면]-> (아이폰:구매)
(아이폰:구매) -[구매하면]-> (에어팟:수령)

행동: 가입, 구매, 사용, 방문, 참여, 등록, 다운로드, 확인, 수령
관계: 가입하면, 구매하면, 사용하면, 방문하면, 참여하면
"""


@tool
def extract_entities_llm(message: str, candidate_entities: str = "") -> str:
    """
    LLM으로 엔티티를 추출하고 DB와 매칭합니다.
    
    Args:
        message: 분석할 MMS 메시지
        candidate_entities: 참고용 후보 엔티티 (콤마 구분, 선택)
    
    Returns:
        JSON 문자열: [{"item_nm": "...", "item_id": "...", "score": 0.95}]
    """
    try:
        # LLM으로 엔티티 추출
        prompt = ENTITY_EXTRACTION_PROMPT.format(message=message)
        if candidate_entities:
            prompt += f"\n\n참고용 후보: {candidate_entities}"
        
        response = _llm.invoke(prompt)
        
        # 응답 파싱
        entities = [e.strip() for e in response.split(',') if e.strip()]
        
        # DB 매칭 (fuzzy search)
        from rapidfuzz import process, fuzz
        
        result = []
        for entity in entities:
            if _extractor.item_pdf.empty:
                continue
            
            choices = _extractor.item_pdf['item_nm'].dropna().astype(str).tolist()
            if not choices:
                continue
            
            matches = process.extract(
                entity,
                choices,
                scorer=fuzz.WRatio,
                limit=3,
                score_cutoff=60
            )
            
            for name, score, idx in matches:
                matched_rows = _extractor.item_pdf[_extractor.item_pdf['item_nm'].astype(str) == name]
                for _, row in matched_rows.head(1).iterrows():
                    result.append({
                        "item_nm": row.get('item_nm', ''),
                        "item_id": row.get('item_id', ''),
                        "item_name_in_msg": entity,
                        "score": score / 100.0
                    })
        
        return json.dumps(result, ensure_ascii=False)
    
    except Exception as e:
        logger.error(f"extract_entities_llm failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def extract_main_info(
    message: str,
    mode: str = "llm",
    context: str = ""
) -> str:
    """
    메시지에서 메인 정보를 추출합니다 (title, purpose, product, channel, sales_script).
    
    Args:
        message: 분석할 MMS 메시지
        mode: 추출 모드 (nlp/rag/llm)
        context: 추가 컨텍스트 (프로그램 정보, 후보 상품 등)
    
    Returns:
        JSON 문자열: {title, purpose, product, channel, sales_script}
    """
    try:
        # 프롬프트 구성
        prompt = MAIN_INFO_PROMPT.format(
            message=message,
            context=context if context else "# 컨텍스트: 없음"
        )
        
        # LLM 호출
        response = _llm.invoke(prompt)
        
        # JSON 추출
        # Try to find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            result = json.loads(json_str)
            return json.dumps(result, ensure_ascii=False)
        else:
            return json.dumps({"error": "No JSON found in response", "raw": response})
    
    except Exception as e:
        logger.error(f"extract_main_info failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def extract_entity_dag(message: str) -> str:
    """
    메시지에서 엔티티 관계 DAG를 추출합니다.
    
    Args:
        message: 분석할 MMS 메시지
    
    Returns:
        JSON 문자열: {"dag": "DAG 텍스트", "entities": ["엔티티1", ...]}
    """
    try:
        prompt = DAG_EXTRACTION_PROMPT.format(message=message)
        response = _llm.invoke(prompt)
        
        # 엔티티 추출
        import re
        entities = set()
        for match in re.finditer(r'\(([^:]+):', response):
            entities.add(match.group(1))
        
        return json.dumps({
            "dag": response,
            "entities": list(entities)
        }, ensure_ascii=False)
    
    except Exception as e:
        logger.error(f"extract_entity_dag failed: {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Test
    test_msg = "갤럭시 Z 플립7 구매하고 5GX 프라임 가입하면 갤럭시 워치 무료"
    
    print("Testing extract_entities_llm...")
    result = extract_entities_llm.invoke({"message": test_msg})
    print(result)
    
    print("\nTesting extract_main_info...")
    result = extract_main_info.invoke({"message": test_msg})
    print(result)
    
    print("\nTesting extract_entity_dag...")
    result = extract_entity_dag.invoke({"message": test_msg})
    print(result)
