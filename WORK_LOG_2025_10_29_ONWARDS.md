# 2025년 10월 29일 이후 작업 내역

작성일: 2025-11-06  
기간: 2025-10-29 ~ 2025-11-06  
총 커밋 수: 31개

---

## 📋 작업 개요

### 주요 영역
1. **info_builder** - 웹 크롤링 및 상품 정보 추출 시스템 개발
2. **mms_extractor_exp** - 엔티티 추출 및 별칭 규칙 처리 로직 개선

---

## 🔧 Part 1: info_builder 개발 (2025-11-03 집중 작업)

### 1.1 상품 크롤러 기본 구조 개발 (11-03 11:33 ~ 12:10)

#### 작업 내용
- **상품 크롤러 초기 구현** (`product_crawler.py`)
  - LLM 기반 상품 정보 추출 기능
  - 상세 페이지 크롤링 기능 (`crawl_details`)
  - 텍스트 청킹 처리 (긴 페이지 분할)
  - Claude 응답 잘림 문제 해결
  - 상품 ID 추출 로직 개선 및 일반화

#### 주요 커밋
- `07cad3d` (11-03 11:33) - Claude 응답 잘림 문제 및 상품 ID 추출 개선
- `545e2aa` (11-03 11:53) - 상품 ID 추출 로직 개선 및 일반화
- `2d3c7da` (11-03 11:58) - LLM 중심 아키텍처로 전환 (규칙 기반 제거)
- `da0fc0b` (11-03 12:07) - 청킹 로직 개선 - 거대 청크 생성 문제 해결
- `d5dca21` (11-03 12:10) - price 정보 제거

#### 생성 파일
- `info_builder/product_crawler.py` - 메인 크롤러 모듈
- `info_builder/config.py` - 설정 파일
- `info_builder/CHUNKING_GUIDE.md` - 텍스트 청킹 가이드
- `info_builder/requirements.txt` - 필수 패키지 목록

---

### 1.2 detail_url 캡처 기능 개발 (11-03 12:25 ~ 15:29)

#### 작업 내용
- **상세 페이지 URL 자동 캡처 기능**
  - 브라우저 실행으로 JavaScript 동적 detail_url 캡처 구현
  - detail_url 자동 생성 기능 추가
  - detail_url 추출 일반화 (특정 사이트 패턴 제거)
  - 다중 selector 패턴 지원 (우선순위 순)
  - Viewport 스크롤 자동화
  - 실패 원인 추적 및 분석

#### 주요 커밋
- `d1f292f` (11-03 12:25) - crawl_details 기능 수정 및 테스트 스크립트 추가
- `2140e1f` (11-03 12:37) - detail_url 자동 생성 기능 추가
- `8ec7d84` (11-03 12:47) - detail_url 추출 일반화 (특정 사이트 패턴 제거)
- `d388fc6` (11-03 15:05) - 브라우저 실행으로 JavaScript 동적 detail_url 캡처 구현
- `c3c5977` (11-03 15:29) - detail_url 캡처 성공률 개선 및 실패 원인 분석
- `45001f4` (11-03 15:29) - detail_url 캡처 개선 내역 문서화

#### 성능 개선
- 초기 성공률: 2/187개 (1%)
- 개선 후: 150+/187개 (80%+)

#### 생성 파일
- `info_builder/product_crawler_example.py` - 사용 예제
- `info_builder/test_crawl_details.py` - 테스트 스크립트
- `info_builder/DETAIL_URL_CAPTURE_IMPROVEMENT.md` - 개선 내역 문서
- `info_builder/DETAIL_URL_ANALYSIS.md` - 원인 분석 문서

---

### 1.3 무한 스크롤 뒤로 가기 문제 해결 (11-03 15:38 ~ 15:54)

#### 문제 상황
- 무한 스크롤 페이지에서 뒤로 가기 시 페이지가 초기 상태로 리셋됨
- 두 번째 상품부터 URL 캡처 실패 (DOM에 로드되지 않음)

#### 해결 방법
- **뒤로 가기 후 재스크롤 구현**
  - 상세 페이지 뒤로 가기 후 무한 스크롤 재실행
  - 마지막 상품은 스크롤 생략 (최적화)
  - 빠른 스크롤 (500ms 간격)

#### 주요 커밋
- `3d6ec9d` (11-03 15:37) - not_found 문제 디버깅 및 테스트 스크립트 버그 수정
- `3532abf` (11-03 15:46) - 디버깅 로그 대폭 강화 및 청크 1개만 처리
- `4a1aadf` (11-03 15:49) - 무한 스크롤 페이지 뒤로 가기 후 재스크롤 구현

#### 성능 영향
- 추가 시간: 상품당 ~3초 (9개 상품 기준 ~24초)
- 성공률: 1/9 (11%) → 9/9 (100%)

#### 생성 파일
- `info_builder/INFINITE_SCROLL_FIX.md` - 문제 해결 문서

---

### 1.4 페이지 타입 자동 감지 기능 (11-03 15:59 ~ 16:15)

#### 작업 내용
- **페이지 타입 자동 감지 및 최적 전략 적용**
  - 4가지 페이지 타입 자동 감지:
    1. 무한 스크롤 (infinite_scroll)
    2. 페이지네이션 (pagination)
    3. 더보기 버튼 (load_more)
    4. 정적 페이지 (static)
  - 타입별 최적 전략 자동 적용
  - 불필요한 재스크롤 자동 생략 (일반 페이지에서 2배 빠름)

#### 주요 커밋
- `bfcc177` (11-03 15:59) - 일반화 개선 계획 문서 작성
- `b6ad57c` (11-03 15:54) - 코드 정리 - page → list_page 변수명 변경
- `92abf54` (11-03 16:07) - 페이지 타입 자동 감지 기능 추가
- `c56a686` (11-03 16:08) - 페이지 타입 자동 감지 요약 문서 추가
- `645e66c` (11-03 16:10) - 예시 파일에 페이지 타입 자동 감지 기능 반영
- `35eb274` (11-03 16:15) - 자동 감지 + 상세 페이지 통합 테스트 추가

#### 성능 개선
- 일반 페이지: 15분 → 7.5분 (50% 시간 절약)
- 불필요한 재스크롤 자동 생략

#### 생성 파일
- `info_builder/page_type_detector.py` - 페이지 타입 감지 모듈
- `info_builder/AUTO_DETECT_GUIDE.md` - 자동 감지 가이드
- `info_builder/PAGE_TYPE_AUTO_DETECT_SUMMARY.md` - 요약 문서
- `info_builder/GENERALIZATION_PLAN.md` - 일반화 개선 계획
- `info_builder/TESTING_AUTO_DETECT_GUIDE.md` - 테스트 가이드
- `info_builder/test_auto_detect_with_details.py` - 통합 테스트 스크립트

#### 추가 문서
- `info_builder/DEBUG_LOG_GUIDE.md` - 디버깅 로그 가이드
- `info_builder/TESTING_STATUS.md` - 테스트 현황 및 가이드

---

### 1.5 웹 크롤러 및 문서화 (11-03 15:02)

#### 작업 내용
- **웹 크롤러 기본 구조 개발**
  - 일반 웹 페이지 크롤링 도구
  - 동적 페이지 크롤링 (JavaScript 렌더링)
  - 무한 스크롤 지원
  - 다양한 출력 형식 지원

#### 주요 커밋
- `f990576` (11-03 15:02) - detail_url 문제 원인 분석 완료

#### 생성 파일
- `info_builder/web_crawler.py` - 웹 크롤러 모듈
- `info_builder/web_crawler_example.py` - 웹 크롤러 예제
- `info_builder/WEB_CRAWLER_GUIDE.md` - 웹 크롤러 가이드
- `info_builder/PRODUCT_CRAWLER_GUIDE.md` - 상품 크롤러 가이드
- `info_builder/ENV_SETUP.md` - 환경 설정 가이드
- `info_builder/QUICKSTART.md` - 빠른 시작 가이드
- `info_builder/TROUBLESHOOTING.md` - 문제 해결 가이드
- `info_builder/README.md` - 전체 프로젝트 README
- `info_builder/check_detail_url.py` - detail_url 체크 스크립트
- `info_builder/example.ipynb` - 예제 노트북

---

## 🔧 Part 2: mms_extractor_exp 개선 (2025-11-04 ~ 11-06)

### 2.1 별칭 규칙 시스템 도입 (11-04 16:04 ~ 11-05 08:47)

#### 작업 내용
- **별칭 규칙 파일 시스템 구축**
  - CSV 기반 별칭 규칙 관리 (`alias_rules.csv`)
  - 별칭 규칙 적용 로직 구현
  - 엔티티 추출 프롬프트 개선

#### 주요 커밋
- `e65b960` (11-04 16:04) - Update mms_extractor, entity_extraction_prompt, and alias_rules
- `35df928` (11-04 18:34) - Update mms_extractor.py, entity_extraction_prompt.py, and alias_rules.csv
- `8acb666` (11-05 08:47) - 별칭 규칙 테스트

#### 변경 파일
- `mms_extractor_exp/data/alias_rules.csv` - 별칭 규칙 데이터 (365줄, 신규 생성)
- `mms_extractor_exp/mms_extractor.py` - 메인 모듈 수정
- `mms_extractor_exp/prompts/entity_extraction_prompt.py` - 프롬프트 개선

---

### 2.2 별칭 규칙 및 엔티티 추출 로직 개선 (11-05 12:51)

#### 작업 내용
- **별칭 규칙 처리 로직 개선**
  - 별칭 규칙 적용 로직 최적화
  - 엔티티 추출 정확도 향상
  - 배치 처리 로직 개선

#### 주요 커밋
- `cf61a13` (11-05 12:51) - feat: 별칭 규칙 및 엔티티 추출 로직 개선

#### 변경 파일
- `mms_extractor_exp/mms_extractor.py` - 44줄 변경
- `mms_extractor_exp/batch.py` - 22줄 추가
- `mms_extractor_exp/data/alias_rules.csv` - 365줄 수정
- `mms_extractor_exp/prompts/entity_extraction_prompt.py` - 2줄 변경
- `mms_extractor_exp/utils.py` - 8줄 변경

---

### 2.3 별칭 규칙 중복 적용 방지 (11-05 18:39 ~ 11-06 10:39)

#### 작업 내용
- **별칭 규칙 중복 적용 방지 로직 개선**
  - 중복 적용 방지 메커니즘 구현
  - 병렬 처리 지원 (`parallel_alias_rule`)
  - 알고리즘 버전 관리

#### 주요 커밋
- `60f935c` (11-05 18:39) - the last version before new alias rule algorithm
- `a88b49d` (11-05 18:53) - parallel_alias_rule
- `b2dc989` (11-06 10:39) - Fix: 별칭 규칙 중복 적용 방지 로직 개선

#### 변경 파일
- `mms_extractor_exp/mms_extractor.py` - 59줄 변경 (중복 방지 로직)
- `mms_extractor_exp/mms_extractor_agentic.ipynb` - 1047줄 추가, 174줄 삭제

---

## 📊 통계 요약

### 커밋 통계
- **총 커밋 수**: 31개
- **주요 작업 기간**: 11월 3일 (하루 집중), 11월 4-6일

### 파일 변경 통계
- **신규 파일**: 약 20개 이상
- **수정 파일**: 약 15개
- **주요 모듈**:
  - `info_builder/product_crawler.py` - 신규 생성 및 다수 수정
  - `mms_extractor_exp/mms_extractor.py` - 다수 수정
  - `mms_extractor_exp/data/alias_rules.csv` - 신규 생성 및 수정

### 코드 라인 수
- **info_builder**: 약 3,000+ 라인 추가
- **mms_extractor_exp**: 약 200+ 라인 수정

---

## 🎯 주요 성과

### info_builder
1. ✅ **완전한 상품 크롤링 시스템 구축**
   - 페이지 타입 자동 감지
   - LLM 기반 정보 추출
   - 상세 페이지 자동 방문
   - 구조화된 데이터 출력 (CSV, JSON, Excel)

2. ✅ **성능 최적화**
   - 일반 페이지: 50% 시간 절약 (재스크롤 생략)
   - detail_url 캡처 성공률: 1% → 80%+

3. ✅ **일반화 가능성**
   - 다양한 웹사이트에 적용 가능한 구조
   - 자동 감지로 수동 설정 최소화

### mms_extractor_exp
1. ✅ **별칭 규칙 시스템 구축**
   - CSV 기반 규칙 관리
   - 중복 적용 방지 로직 구현
   - 병렬 처리 지원

2. ✅ **엔티티 추출 정확도 향상**
   - 별칭 규칙 적용으로 추출 품질 개선
   - 프롬프트 최적화

---

## 📚 생성된 문서

### info_builder 가이드 문서
1. `README.md` - 전체 프로젝트 소개
2. `PRODUCT_CRAWLER_GUIDE.md` - 상품 크롤러 가이드
3. `WEB_CRAWLER_GUIDE.md` - 웹 크롤러 가이드
4. `AUTO_DETECT_GUIDE.md` - 페이지 타입 자동 감지 가이드
5. `PAGE_TYPE_AUTO_DETECT_SUMMARY.md` - 자동 감지 요약
6. `DETAIL_URL_CAPTURE_IMPROVEMENT.md` - detail_url 캡처 개선 내역
7. `INFINITE_SCROLL_FIX.md` - 무한 스크롤 문제 해결
8. `CHUNKING_GUIDE.md` - 텍스트 청킹 가이드
9. `GENERALIZATION_PLAN.md` - 일반화 개선 계획
10. `TESTING_AUTO_DETECT_GUIDE.md` - 테스트 가이드
11. `TESTING_STATUS.md` - 테스트 현황
12. `DEBUG_LOG_GUIDE.md` - 디버깅 로그 가이드
13. `ENV_SETUP.md` - 환경 설정 가이드
14. `QUICKSTART.md` - 빠른 시작 가이드
15. `TROUBLESHOOTING.md` - 문제 해결 가이드

---

## 🔍 세부 변경 내역

### 각 커밋별 상세 변경사항

#### 2025-11-03 (하루 집중 작업)
- **07cad3d** (11:33): info_builder 초기 구조 생성
- **545e2aa** (11:53): 상품 ID 추출 로직 개선
- **2d3c7da** (11:58): LLM 중심 아키텍처 전환
- **da0fc0b** (12:07): 청킹 로직 개선
- **d5dca21** (12:10): price 정보 제거
- **d1f292f** (12:25): crawl_details 기능 수정
- **2140e1f** (12:37): detail_url 자동 생성
- **8ec7d84** (12:47): detail_url 추출 일반화
- **f990576** (15:02): detail_url 문제 원인 분석
- **322d11b** (14:03): 디버깅 로그 추가
- **d388fc6** (15:05): JavaScript 동적 detail_url 캡처
- **c3c5977** (15:29): detail_url 캡처 성공률 개선
- **45001f4** (15:29): detail_url 캡처 개선 내역 문서화
- **3d6ec9d** (15:37): not_found 문제 디버깅
- **6a6ca9b** (15:38): 테스트 현황 문서 작성
- **3532abf** (15:46): 디버깅 로그 강화
- **10131d8** (15:47): 디버깅 로그 가이드 문서
- **4a1aadf** (15:49): 무한 스크롤 재스크롤 구현
- **c1beadd** (15:50): 무한 스크롤 해결 문서
- **b6ad57c** (15:54): 코드 정리
- **bfcc177** (15:59): 일반화 개선 계획 문서
- **92abf54** (16:07): 페이지 타입 자동 감지 기능
- **c56a686** (16:08): 자동 감지 요약 문서
- **645e66c** (16:10): 예시 파일 업데이트
- **35eb274** (16:15): 통합 테스트 추가

#### 2025-11-04
- **e65b960** (16:04): 별칭 규칙 시스템 도입
- **35df928** (18:34): 별칭 규칙 및 프롬프트 업데이트

#### 2025-11-05
- **8acb666** (08:47): 별칭 규칙 테스트
- **cf61a13** (12:51): 별칭 규칙 및 엔티티 추출 로직 개선
- **60f935c** (18:39): 알고리즘 버전 관리
- **a88b49d** (18:53): 병렬 처리 지원

#### 2025-11-06
- **b2dc989** (10:39): 별칭 규칙 중복 적용 방지 로직 개선

---

## 💡 기술적 하이라이트

### 1. 페이지 타입 자동 감지
- 실제 스크롤 테스트로 무한 스크롤 감지
- 다양한 selector 패턴으로 페이지네이션 감지
- 다국어 지원으로 "더보기" 버튼 감지

### 2. 성능 최적화 전략
- 불필요한 재스크롤 자동 생략
- 페이지 타입별 최적 전략 적용
- 병렬 처리 지원 (별칭 규칙)

### 3. 안정성 향상
- 다중 selector fallback 메커니즘
- Viewport 스크롤 자동화
- 강제 클릭 메커니즘 (JavaScript fallback)
- 실패 원인 추적 및 분석

---

## 🚀 향후 개선 계획

### info_builder
1. URL 패턴 학습 기능 (Phase 2)
2. "더보기" 버튼 자동 클릭 (Phase 3)
3. 병렬 처리 (Phase 4)

### mms_extractor_exp
1. 별칭 규칙 자동 생성
2. 엔티티 매칭 정확도 향상
3. 배치 처리 성능 최적화

---

## 📝 참고 사항

- 모든 작업은 Red Hat Enterprise Linux 8.10 서버 환경에서 테스트되었습니다.
- 맥북에서 코드 작성 후 Git 커밋, 리눅스 서버에서 pull & 테스트하는 워크플로우를 따랐습니다.
- 상세한 문서는 각 가이드 파일을 참고하세요.

---

**작성자**: yongwook  
**작성일**: 2025-11-06  
**버전**: 1.0

