# 🎯 프롬프트 관리 가이드 (Prompt Management Guide)

## 📋 개요

MMS 추출기의 모든 LLM 프롬프트는 이제 `prompts` 디렉토리에서 중앙집중식으로 관리됩니다. 
이 가이드는 프롬프트 관리, 수정, 추가에 대한 표준화된 절차를 제공합니다.

## 🏗️ 디렉토리 구조

```
prompts/
├── __init__.py                     # 모든 프롬프트 통합 임포트
├── main_extraction_prompt.py       # 메인 정보 추출 프롬프트
├── entity_extraction_prompt.py     # 엔티티 추출 프롬프트 
├── dag_extraction_prompt.py        # DAG 관계 추출 프롬프트
└── retry_enhancement_prompt.py     # 재시도 및 폴백 프롬프트
```

## 📝 프롬프트 카테고리

### 1. 메인 정보 추출 (`main_extraction_prompt.py`)
- **목적**: MMS 메시지에서 핵심 정보 추출
- **주요 프롬프트**:
  - `build_extraction_prompt()`: 메인 추출 프롬프트 생성
  - `JSON_SCHEMA`: 출력 스키마 정의
  - `CHAIN_OF_THOUGHT_*`: 모드별 사고 과정 프롬프트

### 2. 엔티티 추출 (`entity_extraction_prompt.py`)
- **목적**: 상품명, 서비스명 등 엔티티 추출
- **주요 프롬프트**:
  - `DEFAULT_ENTITY_EXTRACTION_PROMPT`: 기본 엔티티 추출
  - `DETAILED_ENTITY_EXTRACTION_PROMPT`: 상세 엔티티 추출 (이전 settings.py에서 이동)
  - `build_entity_extraction_prompt()`: 엔티티 추출 프롬프트 생성

### 3. DAG 추출 (`dag_extraction_prompt.py`)
- **목적**: 엔티티 간 관계 그래프 추출
- **주요 프롬프트**:
  - `build_dag_extraction_prompt()`: DAG 추출 프롬프트 생성
  - `DAG_EXTRACTION_PROMPT_TEMPLATE`: DAG 템플릿

### 4. 재시도 및 폴백 (`retry_enhancement_prompt.py`)
- **목적**: LLM 응답 오류 시 재시도 및 폴백 처리
- **주요 프롬프트**:
  - `enhance_prompt_for_retry()`: 재시도용 프롬프트 강화
  - `get_fallback_result()`: 기본 폴백 결과

## 🔧 프롬프트 사용 방법

### 코드에서 프롬프트 임포트
```python
from prompts import (
    build_extraction_prompt,
    DETAILED_ENTITY_EXTRACTION_PROMPT,
    build_dag_extraction_prompt,
    enhance_prompt_for_retry
)
```

### 프롬프트 사용 예시
```python
# 엔티티 추출 프롬프트 사용
base_prompt = DETAILED_ENTITY_EXTRACTION_PROMPT
entity_prompt = build_entity_extraction_prompt(message, base_prompt)

# 메인 추출 프롬프트 사용
main_prompt = build_extraction_prompt(message, context, products)
```

## ✏️ 프롬프트 수정 가이드

### 1. 기존 프롬프트 수정
1. 해당 프롬프트 파일 편집 (예: `entity_extraction_prompt.py`)
2. 프롬프트 내용 수정
3. 변경 내역을 Git 커밋에 명시
4. 테스트를 통해 성능 검증

### 2. 새 프롬프트 추가
1. 적절한 카테고리의 파일에 프롬프트 추가
2. `__init__.py`에 임포트 추가:
   ```python
   from .your_module import NEW_PROMPT
   
   __all__ = [
       # ... 기존 항목들
       'NEW_PROMPT',
   ]
   ```
3. 필요시 새로운 모듈 파일 생성

### 3. 프롬프트 모듈 생성
```python
# new_prompt_module.py
"""
새로운 프롬프트 모듈 설명
"""

NEW_PROMPT_TEMPLATE = """
Your new prompt template here...
"""

def build_new_prompt(param1, param2):
    """
    새 프롬프트 생성 함수
    """
    return NEW_PROMPT_TEMPLATE.format(
        param1=param1,
        param2=param2
    )
```

## 🚫 금지사항

### ❌ 하지 말아야 할 것들
1. **settings.py에 프롬프트 추가하지 마세요**
   - 모든 프롬프트는 `prompts` 디렉토리에서 관리
   
2. **하드코딩된 프롬프트 사용 금지**
   ```python
   # ❌ 나쁜 예시
   prompt = "Extract entities from this text..."
   
   # ✅ 좋은 예시
   from prompts import DETAILED_ENTITY_EXTRACTION_PROMPT
   prompt = DETAILED_ENTITY_EXTRACTION_PROMPT
   ```

3. **프롬프트 중복 생성 금지**
   - 기존 프롬프트를 재사용하거나 확장하여 사용

## 🔄 마이그레이션 완료 사항

### ✅ 완료된 작업들
1. **settings.py → prompts 디렉토리**
   - `entity_extraction_prompt` → `DETAILED_ENTITY_EXTRACTION_PROMPT`
   
2. **코드 참조 업데이트**
   - 모든 프롬프트 참조가 `prompts` 모듈로 변경됨
   
3. **하위 호환성**
   - 기존 설정 기반 프롬프트도 여전히 지원 (deprecated)

### 📈 이점
- **중앙집중식 관리**: 모든 프롬프트가 한 곳에서 관리
- **버전 관리**: Git을 통한 프롬프트 변경 이력 추적
- **모듈화**: 기능별로 분리된 프롬프트 관리
- **재사용성**: 다른 프로젝트에서도 쉽게 활용 가능
- **유지보수성**: 프롬프트 수정 시 영향 범위 최소화

## 🔍 문제 해결

### 프롬프트 임포트 오류 시
```python
# 오류 발생 시 확인사항
1. __init__.py에 해당 프롬프트가 임포트되어 있는지 확인
2. 프롬프트 파일명과 함수명이 일치하는지 확인
3. __all__ 리스트에 프롬프트가 포함되어 있는지 확인
```

### 성능 이슈 발생 시
1. 프롬프트 길이 확인 (너무 길면 토큰 제한 초과 가능)
2. 프롬프트 명확성 검토 (모호한 지시사항은 성능 저하 유발)
3. A/B 테스트를 통한 프롬프트 성능 비교

---

**작성자**: MMS 분석팀  
**최종 수정**: 2024-09  
**버전**: 1.0.0
