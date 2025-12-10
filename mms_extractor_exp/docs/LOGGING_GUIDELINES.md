# 로깅 레벨 가이드라인

## 📋 개요

이 문서는 MMS Extractor 프로젝트의 일관된 로깅 관행을 위한 가이드라인을 제공합니다.

---

## 🎯 로깅 레벨 정의

### DEBUG (개발/디버깅용)

**사용 시점**: 상세한 디버깅 정보가 필요할 때

**예시**:
- 프롬프트 전문 내용
- 중간 처리 결과 (DataFrame 크기, 변환 전후 비교)
- 함수 진입/종료 로그
- 변수 값 추적
- 내부 알고리즘 단계별 진행 상황

**코드 예시**:
```python
logger.debug(f"프롬프트 길이: {len(prompt)} 문자")
logger.debug(f"DataFrame 변환: {before_size} -> {after_size}")
logger.debug(f"후보 상품 목록 포함 여부: {'참고용 후보 상품 이름 목록' in rag_context}")
```

**프로덕션 환경**: 일반적으로 비활성화됨 (로그 양 감소)

---

### INFO (운영 정보)

**사용 시점**: 주요 단계 시작/완료, 요약 정보

**예시**:
- Workflow 단계 시작/완료
- 데이터 로드 완료 (건수 포함)
- 주요 설정 값
- 성공적인 처리 완료
- 시스템 상태 변경

**코드 예시**:
```python
logger.info("🚀 엔티티 추출 시작")
logger.info(f"✅ 상품 정보 로드 완료: {len(items)}개")
logger.info(f"사용 LLM 모델: {model_name}")
logger.info(f"처리 완료: messages={count}, time={elapsed:.2f}s")
```

**프로덕션 환경**: 기본 활성화 (주요 동작 추적)

---

### WARNING (주의 필요)

**사용 시점**: 예상 가능한 문제, fallback 사용

**예시**:
- 선택적 데이터 로드 실패 (계속 진행 가능)
- Fallback 메커니즘 사용
- 권장하지 않는 설정 사용
- 성능 저하 가능성
- 데이터 품질 이슈

**코드 예시**:
```python
logger.warning("⚠️ 정지어 파일 로드 실패, 빈 리스트 사용")
logger.warning("Fallback 결과 반환")
logger.warning(f"텍스트가 너무 깁니다 ({len(text)} 문자). 처음 10000자만 사용합니다.")
```

**프로덕션 환경**: 활성화 (잠재적 문제 모니터링)

---

### ERROR (오류 발생)

**사용 시점**: 실제 오류 발생, 처리 실패

**예시**:
- 필수 데이터 로드 실패
- LLM 호출 실패 (재시도 후에도)
- 파싱 오류
- 예상치 못한 예외
- 시스템 장애

**코드 예시**:
```python
logger.error(f"❌ 필수 데이터 로드 실패: {e}")
logger.error(f"상세 오류: {traceback.format_exc()}")
logger.error(f"LLM 호출 최종 실패: {e}")
```

**프로덕션 환경**: 항상 활성화 (즉시 대응 필요)

---

## 🎨 적용 규칙

### 1. 이모지 사용

INFO 레벨 이상에서 시각적 구분을 위해 사용:

- 🚀 **시작**: 주요 프로세스 시작
- ✅ **성공**: 작업 완료
- ⚠️ **경고**: 주의 필요
- ❌ **오류**: 실패/에러
- 🔍 **분석**: 데이터 분석 중
- 📊 **통계**: 통계 정보
- 🔗 **연결**: 외부 시스템 연결
- 💾 **저장**: 데이터 저장
- 🎯 **목표**: 목표 달성

### 2. 구조화된 로그

중요 정보는 key=value 형식 사용:

```python
# ✅ 좋은 예
logger.info(f"처리 완료: messages={count}, time={elapsed:.2f}s, success_rate={rate:.1f}%")

# ❌ 나쁜 예
logger.info(f"처리 완료했습니다. {count}개 메시지를 {elapsed:.2f}초에 처리했고 성공률은 {rate:.1f}%입니다.")
```

### 3. 민감 정보 제외

API 키, 개인정보, 비밀번호 등은 로그에 포함하지 않음:

```python
# ✅ 좋은 예
logger.info(f"API 호출 성공: endpoint={endpoint}")

# ❌ 나쁜 예
logger.info(f"API 호출 성공: api_key={api_key}, endpoint={endpoint}")
```

### 4. 성능 고려

DEBUG 로그는 프로덕션에서 비활성화 가능하도록 작성:

```python
# ✅ 좋은 예 - 조건부 로깅
if logger.isEnabledFor(logging.DEBUG):
    expensive_debug_info = calculate_expensive_debug_info()
    logger.debug(f"상세 정보: {expensive_debug_info}")

# ❌ 나쁜 예 - 항상 계산
logger.debug(f"상세 정보: {calculate_expensive_debug_info()}")
```

### 5. 예외 로깅

예외 발생 시 traceback 포함:

```python
try:
    risky_operation()
except Exception as e:
    logger.error(f"작업 실패: {e}")
    logger.error(f"상세 오류: {traceback.format_exc()}")
```

---

## 📝 실전 예시

### Workflow 단계 로깅

```python
class EntityExtractionStep(WorkflowStep):
    def execute(self, state: WorkflowState) -> WorkflowState:
        logger.info("🚀 엔티티 추출 단계 시작")  # INFO: 단계 시작
        
        try:
            entities = self.extract_entities(state.msg)
            logger.debug(f"추출된 엔티티: {entities}")  # DEBUG: 상세 결과
            logger.info(f"✅ 엔티티 추출 완료: {len(entities)}개")  # INFO: 요약
            
        except Exception as e:
            logger.error(f"❌ 엔티티 추출 실패: {e}")  # ERROR: 실패
            logger.error(f"상세 오류: {traceback.format_exc()}")
            raise
        
        return state
```

### 데이터 로딩 로깅

```python
def load_data(self):
    logger.info("📊 데이터 로딩 시작")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✅ 데이터 로드 완료: {len(df)}개 행")
        
        if df.empty:
            logger.warning("⚠️ 데이터가 비어있습니다")
        
        logger.debug(f"컬럼: {list(df.columns)}")
        logger.debug(f"샘플 데이터:\n{df.head()}")
        
    except FileNotFoundError:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        raise
```

### LLM 호출 로깅

```python
def call_llm(self, prompt):
    logger.info("🔗 LLM 호출 시작")
    logger.debug(f"프롬프트 길이: {len(prompt)} 문자")
    
    try:
        response = self.llm.invoke(prompt)
        logger.info(f"✅ LLM 응답 수신: {len(response.content)} 문자")
        logger.debug(f"응답 내용: {response.content[:200]}...")
        
    except Exception as e:
        logger.error(f"❌ LLM 호출 실패: {e}")
        logger.warning("⚠️ Fallback 메커니즘 사용")
        response = self.get_fallback_response()
    
    return response
```

---

## 🔧 로깅 설정

### 개발 환경

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,  # 모든 레벨 출력
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 프로덕션 환경

```python
import logging

logging.basicConfig(
    level=logging.INFO,  # INFO 이상만 출력
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### 환경별 동적 설정

```python
import os
import logging

log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level))
```

---

## ✅ 체크리스트

새로운 로깅 추가 시 확인사항:

- [ ] 적절한 로깅 레벨 선택 (DEBUG/INFO/WARNING/ERROR)
- [ ] 이모지 사용 (INFO 이상)
- [ ] 구조화된 형식 (key=value)
- [ ] 민감 정보 제외
- [ ] 예외 발생 시 traceback 포함
- [ ] DEBUG 로그는 성능 고려

---

*작성일: 2025-12-10*  
*대상: mms_extractor_exp 프로젝트*  
*버전: 1.0*
