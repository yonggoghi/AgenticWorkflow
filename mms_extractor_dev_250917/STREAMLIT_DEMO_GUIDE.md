# Streamlit Demo 실행 가이드

## 문제 해결

기존에 `python demo_streamlit.py -- --api-port 8000 --demo-port 8082` 명령어가 작동하지 않던 문제를 해결했습니다.

## 해결 방법

### 방법 1: 실행 스크립트 사용 (권장)

```bash
cd mms_extractor_unified
./run_demo_streamlit.sh --api-port 8000 --demo-port 8082
```

### 방법 2: 직접 Python 실행

```bash
cd mms_extractor_unified
python demo_streamlit.py --api-port 8000 --demo-port 8082
```

### 방법 3: Streamlit 명령어 사용

```bash
cd mms_extractor_unified
streamlit run demo_streamlit.py -- --api-port 8000 --demo-port 8082
```

## 옵션 설명

- `--api-port`: MMS Extractor API 서버 포트 (기본값: 8000)
- `--demo-port`: Demo 서버 포트 (기본값: 8082)

## 수정 사항

1. **명령행 인수 파싱 개선**: Streamlit의 내부 인수들과 충돌하지 않도록 커스텀 파싱 로직 구현
2. **실행 스크립트 추가**: `run_demo_streamlit.sh` 스크립트로 간편한 실행 지원
3. **에러 처리 강화**: 잘못된 인수 값에 대한 기본값 처리

## 확인 사항

실행 전에 다음 사항을 확인하세요:

1. **API 서버 실행**: MMS Extractor API가 지정된 포트에서 실행 중인지 확인
2. **Demo 서버 실행**: Demo 서버가 지정된 포트에서 실행 중인지 확인
3. **포트 충돌**: 지정된 포트들이 다른 프로세스에서 사용 중이지 않은지 확인

## 예시

```bash
# 기본 포트로 실행
./run_demo_streamlit.sh

# 커스텀 포트로 실행
./run_demo_streamlit.sh --api-port 9000 --demo-port 9001

# 도움말 보기
./run_demo_streamlit.sh --help
```

## 트러블슈팅

### 1. Permission denied 오류
```bash
chmod +x run_demo_streamlit.sh
```

### 2. 포트 충돌 오류
```bash
# 포트 사용 확인
lsof -i :8000
lsof -i :8082

# 프로세스 종료
kill -9 <PID>
```

### 3. 모듈 import 오류
```bash
# 필요한 패키지 설치
pip install streamlit requests pandas
```
