# MMS Agent 설치 및 설정 가이드

## 필수 의존성 설치

### 1. 기본 패키지 설치

venv를 활성화한 상태에서:

```bash
pip install kiwipiepy sentence-transformers
```

또는 개별 설치:
```bash
pip install kiwipiepy
pip install sentence-transformers  # torch도 자동 설치됨
```

### 2. 테스트 실행

```bash
python -m mms_agent.tests.test_nonllm_tools
```

## 현재 상태

- ✅ 의존성 분리 완료 (mms_extractor_exp와 독립)
- ✅ 인코딩 자동 감지 (cp949/euc-kr/utf-8)
- ⏳ 패키지 설치 대기 중

## 다음 단계

패키지 설치 후:
- Phase 1 Week 2: LLM 기반 도구 구현
- Agent 구성
