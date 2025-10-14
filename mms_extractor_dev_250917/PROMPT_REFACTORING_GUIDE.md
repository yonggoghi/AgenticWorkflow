# 프롬프트 리팩토링 가이드

## 개요

MMS Extractor의 프롬프트들이 별도 모듈로 분리되어 코드의 가독성과 유지보수성이 향상되었습니다.

## 변경 사항

### 🔄 기존 구조
- 프롬프트가 Python 코드 내에 하드코딩되어 있음
- 긴 프롬프트 문자열로 인한 코드 가독성 저하
- 프롬프트 수정 시 코드 파일 직접 편집 필요

### ✨ 새로운 구조
- 프롬프트가 별도 모듈(`prompts/`)로 분리
- 구조화된 프롬프트 템플릿 시스템
- 중앙화된 프롬프트 관리

## 프롬프트 모듈 구조

```
prompts/
├── __init__.py                     # 모듈 초기화 및 임포트
├── main_extraction_prompt.py       # 메인 정보 추출 프롬프트
├── retry_enhancement_prompt.py     # 재시도 및 강화 프롬프트
├── dag_extraction_prompt.py        # DAG 추출 프롬프트
└── entity_extraction_prompt.py     # 엔티티 추출 프롬프트
```

### 1. 메인 추출 프롬프트 (`main_extraction_prompt.py`)
- **기능**: 광고 메시지에서 제목, 목적, 상품, 채널, 프로그램 정보 추출
- **주요 구성요소**:
  - 모드별 사고 과정 템플릿
  - JSON 스키마 정의
  - 추출 가이드라인
  - 일관성 유지 지침

### 2. 재시도 강화 프롬프트 (`retry_enhancement_prompt.py`)
- **기능**: LLM 호출 실패 시 프롬프트 강화 및 fallback 처리
- **주요 구성요소**:
  - 스키마 응답 방지 지시사항
  - Fallback 결과 템플릿

### 3. DAG 추출 프롬프트 (`dag_extraction_prompt.py`)
- **기능**: 엔티티 간 관계를 DAG 형태로 추출
- **주요 구성요소**:
  - 7단계 분석 프로세스
  - DAG 구성 전략
  - 출력 형식 가이드라인

### 4. 엔티티 추출 프롬프트 (`entity_extraction_prompt.py`)
- **기능**: NLP 기반 엔티티 추출
- **주요 구성요소**:
  - 기본 엔티티 추출 프롬프트
  - 후보 엔티티 포함 템플릿

## 사용법

### 기본 사용
```python
from prompts import build_extraction_prompt, enhance_prompt_for_retry

# 메인 추출 프롬프트 생성
prompt = build_extraction_prompt(
    message="광고 메시지",
    rag_context="RAG 컨텍스트",
    product_info_extraction_mode="llm"
)

# 재시도 프롬프트 강화
enhanced_prompt = enhance_prompt_for_retry(original_prompt)
```

### 고급 사용
```python
from prompts import (
    build_dag_extraction_prompt,
    build_entity_extraction_prompt,
    get_fallback_result
)

# DAG 추출 프롬프트
dag_prompt = build_dag_extraction_prompt("분석할 메시지")

# 엔티티 추출 프롬프트
entity_prompt = build_entity_extraction_prompt("메시지", "기본 프롬프트")

# Fallback 결과
fallback = get_fallback_result()
```

## 이점

### 🎯 유지보수성 향상
- 프롬프트 수정 시 코드 변경 없이 프롬프트 파일만 수정
- 프롬프트별 독립적인 관리 가능

### 📖 가독성 개선
- 메인 코드에서 비즈니스 로직에 집중 가능
- 프롬프트 구조와 내용의 명확한 분리

### 🔄 재사용성
- 다른 프로젝트에서 프롬프트 모듈 재사용 가능
- 표준화된 프롬프트 인터페이스

### 🧪 테스트 용이성
- 프롬프트별 독립적인 테스트 가능
- A/B 테스트를 위한 프롬프트 변형 용이

## 마이그레이션 가이드

### 기존 코드에서 프롬프트 분리하기

1. **프롬프트 식별**
   ```python
   # 기존 코드
   prompt = f"""
   Extract information from: {message}
   Return JSON format...
   """
   ```

2. **프롬프트 모듈로 이동**
   ```python
   # prompts/my_prompt.py
   TEMPLATE = """
   Extract information from: {message}
   Return JSON format...
   """
   
   def build_my_prompt(message):
       return TEMPLATE.format(message=message)
   ```

3. **메인 코드 업데이트**
   ```python
   # 기존 코드 대신
   from prompts import build_my_prompt
   prompt = build_my_prompt(message)
   ```

## 모범 사례

### 1. 프롬프트 구조화
- 템플릿 상수와 함수 분리
- 명확한 함수명과 docstring 사용
- 매개변수 타입 힌트 제공

### 2. 버전 관리
- 프롬프트 변경 시 버전 기록
- 이전 버전과의 호환성 고려

### 3. 문서화
- 각 프롬프트의 목적과 사용법 문서화
- 예제 코드 제공

## 향후 개선 계획

### 1. 동적 프롬프트 로딩
- 설정 파일에서 프롬프트 경로 지정
- 런타임 프롬프트 변경 지원

### 2. 프롬프트 템플릿 엔진
- Jinja2 등 템플릿 엔진 도입
- 더 복잡한 프롬프트 로직 지원

### 3. 프롬프트 성능 모니터링
- 프롬프트별 성능 지표 수집
- A/B 테스트 자동화

---

이 리팩토링을 통해 MMS Extractor는 더욱 유지보수하기 쉽고 확장 가능한 구조가 되었습니다.
