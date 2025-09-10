# MMS Extractor - Production Version

이 디렉토리는 MMS Extractor의 프로덕션 배포용 버전입니다.

## 주요 구성 요소

### 핵심 실행 파일
- `mms_extractor.py` - 메인 MMS 추출기 CLI
- `api.py` - REST API 서버
- `batch.py` - 배치 처리 스크립트
- `entity_dag_extractor.py` - 엔티티 DAG 추출기
- `utils.py` - 유틸리티 함수들

### 설정 및 데이터
- `config/` - 시스템 설정 파일들
- `data/` - 필수 데이터 파일들 (상품 정보, 별칭 규칙 등)
- `models/` - 임베딩 모델 파일들
- `logs/` - 로그 파일 저장 디렉토리
- `prompts/` - 구조화된 프롬프트 템플릿들

### 문서
- `README_DATABASE.md` - 데이터베이스 연동 가이드
- `USAGE_GUIDE.md` - 상세 사용법 가이드
- `API_MULTIPROCESS_GUIDE.md` - API 멀티프로세스 처리 가이드
- `BATCH_PARALLEL_GUIDE.md` - 배치 병렬 처리 가이드
- `PROMPT_REFACTORING_GUIDE.md` - 프롬프트 리팩토링 가이드

## 빠른 시작

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# 환경 변수 설정 (.env 파일 생성)
cp .env.example .env
# .env 파일을 편집하여 API 키 등을 설정하세요
```

### 2. 기본 사용법

#### CLI 사용
```bash
python mms_extractor.py --message "광고 메시지 텍스트"
```

#### API 서버 실행
```bash
python api.py --host 0.0.0.0 --port 8000
```

#### 배치 처리
```bash
python batch.py --batch-size 20
```

## 상세 문서

각 기능에 대한 자세한 사용법은 해당 가이드 문서를 참조하세요:

- **전체 사용법**: `USAGE_GUIDE.md`
- **데이터베이스 연동**: `README_DATABASE.md`
- **API 멀티프로세스**: `API_MULTIPROCESS_GUIDE.md`
- **배치 병렬 처리**: `BATCH_PARALLEL_GUIDE.md`

## 지원

- 기술 문의: 개발팀
- 버그 리포트: GitHub Issues

---

**버전**: Production v1.0
**마지막 업데이트**: 2025년 9월
