# MMS Agent

Agent-based MMS (광고 메시지) 추출 시스템

## 📋 개요

`mms_agent`는 MMS 광고 메시지에서 정보를 추출하는 Agent 기반 시스템입니다. 기존 `mms_extractor_exp`의 견고한 로직을 독립적인 도구(Tools)로 분리하여, LangChain Agent가 상황에 맞게 선택하고 활용할 수 있도록 설계되었습니다.

### 핵심 특징

- ✅ **완전 독립적**: `mms_extractor_exp`와 런타임 의존성 분리
- ✅ **8개 도구**: 5개 Non-LLM + 3개 LLM 도구
- ✅ **Graceful Degradation**: Optional 의존성 자동 처리
- ✅ **실전 검증**: 실제 MMS 광고로 테스트 완료

## 📂 구조

```
mms_agent/
├── __init__.py
├── core/                    # 핵심 기능
│   ├── data_loader.py       # 데이터 로딩 (CSV)
│   ├── extractor_base.py    # 기본 추출 로직
│   └── llm_client.py        # LLM 클라이언트
├── tools/                   # LangChain Tools
│   ├── entity_tools.py      # 엔티티 추출 도구
│   ├── classification_tools.py  # 분류 도구
│   ├── matching_tools.py    # 매칭 도구
│   └── llm_tools.py         # LLM 기반 도구
├── agents/                  # Agent 구현 (Phase 2)
└── tests/                   # 테스트
    ├── test_nonllm_tools.py
    └── test_llm_tools.py
```

## 🛠️ 도구 목록

### Non-LLM 도구 (5개)

#### 1. `search_entities_kiwi`
Kiwi 형태소 분석으로 엔티티 추출
```python
from mms_agent.tools import search_entities_kiwi

result = search_entities_kiwi.invoke({
    "message": "갤럭시 Z 플립7 구매하고 5GX 프라임 가입"
})
# Returns: {
#   "entities": ["갤럭시", "Z", "플립", "구매", "5GX", "프라임", "가입"],
#   "candidate_items": ["갤럭시 Z플립6", ...],
#   "extra_item_count": 120
# }
```

#### 2. `search_entities_fuzzy`
Fuzzy matching으로 상품 DB 검색
```python
from mms_agent.tools import search_entities_fuzzy

result = search_entities_fuzzy.invoke({
    "entities": "갤럭시,아이폰,넷플릭스",
    "threshold": 0.5
})
# Returns: JSON string with matched items and scores
```

#### 3. `classify_program`
임베딩 유사도 기반 프로그램 분류
```python
from mms_agent.tools import classify_program

result = classify_program.invoke({
    "message": "5GX 프라임 요금제 가입",
    "top_k": 5
})
# Returns: {
#   "programs": [{"pgm_nm": "...", "similarity": 0.95}, ...],
#   "context": "프로그램명 : 키워드\n..."
# }
```

#### 4. `match_store_info`
대리점명으로 조직 정보 검색
```python
from mms_agent.tools import match_store_info

result = match_store_info.invoke({
    "store_name": "새서울대리점 대치직영점"
})
# Returns: JSON string with org info and codes
```

#### 5. `validate_entities`
별칭 규칙 기반 엔티티 검증
```python
from mms_agent.tools import validate_entities
import json

result = validate_entities.invoke({
    "entities_json": json.dumps([{"item_nm": "아이폰"}]),
    "message": "원본 메시지"
})
# Returns: JSON string with validated entities
```

### LLM 도구 (3개)

#### 6. `extract_entities_llm`
LLM으로 엔티티 추출 + DB 매칭
```python
from mms_agent.tools import extract_entities_llm

result = extract_entities_llm.invoke({
    "message": "갤럭시 Z 플립7 구매",
    "candidate_entities": ""  # Optional
})
# Returns: JSON with extracted and matched entities
```

#### 7. `extract_main_info`
메인 정보 추출 (title, purpose, product, channel, sales_script)
```python
from mms_agent.tools import extract_main_info

result = extract_main_info.invoke({
    "message": "MMS 광고 메시지",
    "mode": "llm",  # or "rag", "nlp"
    "context": ""   # Optional: program info, candidates
})
# Returns: JSON with {title, purpose, product, channel, sales_script}
```

#### 8. `extract_entity_dag`
엔티티 관계 DAG 추출
```python
from mms_agent.tools import extract_entity_dag

result = extract_entity_dag.invoke({
    "message": "MMS 광고 메시지"
})
# Returns: JSON with {"dag": "...", "entities": [...]}
```

## 🚀 설치 및 설정

### 1. 의존성 설치

```bash
# venv 활성화
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 필수 패키지
pip install kiwipiepy sentence-transformers

# LangChain (도구 사용)
pip install langchain langchain-openai

# 기타
pip install pandas rapidfuzz
```

### 2. 환경 설정

프로젝트 루트에 `.env` 파일 생성:

```bash
# LLM API 설정 (기존 mms_extractor_exp 설정 재사용)
CUSTOM_API_KEY=your_api_key_here
CUSTOM_BASE_URL=https://api.platform.a15t.com/v1
LLM_MODEL=skt/ax4

# 또는 mms_extractor_exp/.env 복사
cp mms_extractor_exp/.env .
```

### 3. 데이터 파일

데이터는 `mms_extractor_exp/data/`에서 자동으로 로드됩니다:
- `offer_master_data.csv` - 상품 정보
- `org_info_all_250605.csv` - 조직/대리점 정보
- `pgm_tag_ext_250516.csv` - 프로그램 분류
- `alias_rules.csv` - 별칭 규칙

## 🧪 테스트

### Non-LLM 도구 테스트
```bash
python -m mms_agent.tests.test_nonllm_tools
```

### LLM 도구 테스트 (API 키 필요)
```bash
python -m mms_agent.tests.test_llm_tools
```

## 📝 사용 예시

### 전체 파이프라인
```python
from mms_agent.tools import (
    search_entities_kiwi,
    classify_program,
    extract_main_info,
    match_store_info
)
import json

message = """새서울대리점 대치직영점에서 갤럭시 Z 플립7 구매하고 
5GX 프라임 요금제 가입하면 갤럭시 워치 무료 증정"""

# 1. 엔티티 추출
entities = search_entities_kiwi.invoke({"message": message})
print(f"Entities: {entities['entities']}")

# 2. 프로그램 분류
programs = classify_program.invoke({"message": message, "top_k": 3})
print(f"Top program: {programs['programs'][0]['pgm_nm']}")

# 3. 메인 정보 추출
info = extract_main_info.invoke({
    "message": message,
    "mode": "llm",
    "context": programs['context']
})
result = json.loads(info)
print(f"Title: {result['title']}")
print(f"Products: {[p['name'] for p in result['product']]}")

# 4. 대리점 매칭
stores = match_store_info.invoke({"store_name": "새서울대리점"})
print(f"Matched stores: {json.loads(stores)}")
```

## 🔄 의존성 관리

### Optional 의존성
시스템은 다음 패키지가 없어도 작동합니다:

- **Kiwi 없음** → `search_entities_kiwi` 비활성화
- **SentenceTransformers/Torch 없음** → `classify_program` 비활성화
- **LLM API 없음** → LLM 도구들 에러 반환

### 완전 독립성
- ✅ `mms_extractor_exp`와 런타임 분리
- ✅ 데이터만 공유 (`mms_extractor_exp/data/`)
- ✅ LLM 설정만 재사용 (`config/settings.py`)

## 🎯 다음 단계

### Phase 2: Agent 구성
- [ ] EntityExtractionAgent 구현
- [ ] MainExtractionAgent 구현
- [ ] Full Agent 통합

### Phase 3: 검증 및 최적화
- [ ] A/B 테스트 (기존 vs Agent)
- [ ] 성능 벤치마크
- [ ] 프롬프트 튜닝

## 📊 성능

**테스트 결과** (실제 MMS 광고 메시지):

| 도구 | 상태 | 특징 |
|------|------|------|
| search_entities_kiwi | ✅ | 형태소 분석, 320개 후보 추출 |
| search_entities_fuzzy | ✅ | Fuzzy 매칭, 0.5+ threshold |
| classify_program | ✅ | Top-3 프로그램 분류 |
| match_store_info | ✅ | 5개 매장 매칭 |
| validate_entities | ✅ | 별칭 규칙 검증 |
| extract_entities_llm | ✅ | 18개 엔티티 + DB 매칭 |
| extract_main_info | ✅ | 6개 필드 추출 완료 |
| extract_entity_dag | ✅ | DAG 14개 노드 추출 |

## 📚 관련 문서

- [INSTALL.md](INSTALL.md) - 설치 가이드
- [implementation_plan.md](../../.gemini/antigravity/brain/ee6a68bb-0626-49e6-8286-2953f1bf77fd/implementation_plan.md) - Agent Framework 적용 계획

## 🤝 기여

이 프로젝트는 기존 `mms_extractor_exp`의 검증된 로직을 Agent 패턴으로 리팩토링한 것입니다.

## 📄 라이선스

Internal use only
