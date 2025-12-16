"""
MMS 추출기 프롬프트 모듈 (Prompts Module)
===============================================

🎯 목적
-------
MMS 추출기에서 사용되는 모든 LLM 프롬프트를 외부 모듈로 분리하여
유지보수성과 재사용성을 향상시키는 모듈입니다.

📚 모듈 구성
-----------
- **main_extraction_prompt**: 메인 정보 추출 프롬프트 (상품, 채널, 프로그램)
- **retry_enhancement_prompt**: 재시도 및 폴백 프롬프트
- **dag_extraction_prompt**: DAG 관계 추출 프롬프트
- **entity_extraction_prompt**: 엔티티 추출 프롬프트 (3가지 모드)

🔧 주요 특징
-----------
- **모듈화된 설계**: 각 기능별로 독립적인 파일 관리
- **중앙집중식 임포트**: `__init__.py`를 통한 통합 접근
- **버전 관리**: 프롬프트 변경 사항 추적 가능
- **재사용성**: 다른 프로젝트에서도 쉽게 활용 가능

📝 사용 예시
-----------
```python
# 1. 모든 프롬프트 함수 임포트
from prompts import (
    build_extraction_prompt,
    build_dag_extraction_prompt,
    enhance_prompt_for_retry,
    get_fallback_result,
    build_entity_extraction_prompt
)

# 2. 메인 추출 프롬프트 사용
prompt = build_extraction_prompt(
    message="아이폰 17 구매 시 캐시백 제공",
    context="RAG 컨텍스트...",
    products=["아이폰 17"]
)

# 3. DAG 추출 프롬프트 사용
dag_prompt = build_dag_extraction_prompt(message)

# 4. 엔티티 추출 프롬프트 사용 (컨텍스트 모드별)
from prompts import (
    HYBRID_DAG_EXTRACTION_PROMPT,
    HYBRID_PAIRING_EXTRACTION_PROMPT,
    SIMPLE_ENTITY_EXTRACTION_PROMPT
)

# DAG 모드
entity_prompt = f"{HYBRID_DAG_EXTRACTION_PROMPT}\n\n## message:\n{message}"

# 5. 재시도 프롬프트 사용
enhanced_prompt = enhance_prompt_for_retry(
    original_prompt=prompt,
    previous_response="잘못된 응답",
    error_message="JSON 파싱 실패"
)
```

🏗️ 프롬프트 아키텍처
-------------------

```mermaid
graph TB
    A[MMS Message] --> B{Extraction Type}
    B -->|Main| C[build_extraction_prompt]
    B -->|Entity| D[build_entity_extraction_prompt]
    B -->|DAG| E[build_dag_extraction_prompt]
    
    C --> F[LLM]
    D --> F
    E --> F
    
    F -->|Success| G[Parse Result]
    F -->|Failure| H[enhance_prompt_for_retry]
    H --> F
    
    G -->|Parse Error| I[get_fallback_result]
    G -->|Success| J[Final Result]
    I --> J
```

🔄 업데이트 가이드
--------------
1. **프롬프트 수정 시**: 해당 모듈 파일만 편집
2. **새로운 프롬프트 추가 시**: `__init__.py`에 임포트 추가
3. **버전 관리**: 변경 내역 문서화 (git commit message)

📋 내보내기 목록
--------------
### 메인 정보 추출
- `build_extraction_prompt`: 주요 추출 프롬프트 생성
- `JSON_SCHEMA`: JSON 스키마 정의
- `CHAIN_OF_THOUGHT_LLM_MODE`: LLM 모드 CoT 프롬프트
- `CHAIN_OF_THOUGHT_DEFAULT_MODE`: 기본 모드 CoT 프롬프트
- `CHAIN_OF_THOUGHT_NLP_MODE`: NLP 모드 CoT 프롬프트

### 재시도 및 폴백
- `enhance_prompt_for_retry`: 재시도용 프롬프트 강화
- `get_fallback_result`: 기본 폴백 결과 생성
- `SCHEMA_PREVENTION_INSTRUCTION`: 스키마 반환 방지 지시

### DAG 관계 추출
- `build_dag_extraction_prompt`: DAG 추출 프롬프트 생성
- `DAG_EXTRACTION_PROMPT_TEMPLATE`: DAG 추출 템플릿

### 엔티티 추출
- `build_entity_extraction_prompt`: 엔티티 추출 프롬프트 생성
- `DEFAULT_ENTITY_EXTRACTION_PROMPT`: 기본 엔티티 추출 프롬프트
- `HYBRID_DAG_EXTRACTION_PROMPT`: DAG 모드 프롬프트
- `HYBRID_PAIRING_EXTRACTION_PROMPT`: PAIRING 모드 프롬프트
- `SIMPLE_ENTITY_EXTRACTION_PROMPT`: SIMPLE 모드 프롬프트
- `build_context_based_entity_extraction_prompt`: 동적 프롬프트 생성

"""

from .main_extraction_prompt import (
    build_extraction_prompt,
    JSON_SCHEMA,
    CHAIN_OF_THOUGHT_LLM_MODE,
    CHAIN_OF_THOUGHT_DEFAULT_MODE,
    CHAIN_OF_THOUGHT_NLP_MODE
)

from .retry_enhancement_prompt import (
    enhance_prompt_for_retry,
    get_fallback_result,
    SCHEMA_PREVENTION_INSTRUCTION
)

from .dag_extraction_prompt import (
    build_dag_extraction_prompt,
    DAG_EXTRACTION_PROMPT_TEMPLATE
)

from .entity_extraction_prompt import (
    build_entity_extraction_prompt,
    DEFAULT_ENTITY_EXTRACTION_PROMPT
)

# 외부에서 사용 가능한 모든 함수와 상수들을 명시적으로 정의
__all__ = [
    # 메인 정보 추출 관련
    'build_extraction_prompt',           # 주요 추출 프롬프트 생성
    'JSON_SCHEMA',                      # JSON 스키마 정의
    'CHAIN_OF_THOUGHT_LLM_MODE',        # LLM 모드 CoT 프롬프트
    'CHAIN_OF_THOUGHT_DEFAULT_MODE',    # 기본 모드 CoT 프롬프트
    'CHAIN_OF_THOUGHT_NLP_MODE',        # NLP 모드 CoT 프롬프트
    
    # 재시도 및 폴백 처리 관련
    'enhance_prompt_for_retry',         # 재시도용 프롬프트 강화
    'get_fallback_result',              # 기본 폴백 결과 생성
    'SCHEMA_PREVENTION_INSTRUCTION',    # 스키마 반환 방지 지시
    
    # DAG 관계 추출 관련
    'build_dag_extraction_prompt',      # DAG 추출 프롬프트 생성
    'DAG_EXTRACTION_PROMPT_TEMPLATE',   # DAG 추출 템플릿
    
    # 엔티티 추출 관련
    'build_entity_extraction_prompt',   # 엔티티 추출 프롬프트 생성
    'DEFAULT_ENTITY_EXTRACTION_PROMPT', # 기본 엔티티 추출 프롬프트
]
