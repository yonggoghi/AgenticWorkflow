# 메가스터디 러셀 단과 접수 자동화 매크로

메가스터디 러셀 단과 웹사이트에서 자동으로 로그인하고 특정 강의의 '접수 대기' 버튼을 클릭하는 웹 브라우저 자동화 프로그램입니다.

## 🎯 주요 기능

1. **자동 로그인** - 메가스터디 러셀 계정 자동 로그인 (환경 변수/대화형 지원)
2. **자동 사이트 방문** - 메가스터디 러셀 단과 페이지 자동 접속
3. **영어 탭 클릭** - 과목 탭에서 영어 탭 자동 선택
4. **접수 대기 버튼 클릭** - 특정 강사/강의의 접수 대기 버튼 자동 클릭
5. **🚀 실시간 수강신청 모드** - 특정 시간(예: 19:00)까지 대기 후 초고속 연속 클릭 (1ms 간격) 🆕
6. **팝업 자동 처리** - 접수 대기 팝업 자동 감지 및 스크린샷 저장
7. **스크린샷 저장** - 각 단계별 스크린샷 자동 저장
8. **유연한 설정** - 헤드리스 모드, 속도 조절 등 다양한 옵션 지원

## 📦 설치

### 1. Playwright 설치 (최초 1회만)

```bash
# Playwright 설치
pip install playwright

# 브라우저 설치 (Chromium)
playwright install chromium
```

### 2. 의존성 설치

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/russel_macro
pip install -r requirements.txt
```

## 🚀 빠른 시작

### 1단계: 로그인 정보 설정

#### 방법 1: 환경 변수 (권장)

```bash
export RUSSEL_USERNAME="아이디"
export RUSSEL_PASSWORD="비밀번호"
```

#### 방법 2: 대화형 입력

매크로 실행 시 `--interactive` 옵션 사용

### 2단계: 매크로 실행

```bash
# 방법 1: 환경 변수 사용 (기본 모드)
python russel_macro.py

# 방법 2: 대화형 입력
python russel_macro.py --interactive

# 방법 3: 🚀 실시간 수강신청 모드 (권장: 미리 시작) 🆕
# 서버 시간 차이를 고려하여 몇 분 일찍 시작 (예: 18:58부터 클릭)
python russel_macro.py --rapid-mode --target-time "18:58"

# 또는 즉시 시작 (프로그램 실행 즉시 클릭 시작)
python russel_macro.py --rapid-mode --start-immediately

# 방법 4: 브라우저를 보면서 실행 (디버깅)
python russel_macro.py --slow-mo 1000
```

## 📖 상세 가이드

- **빠른 시작 가이드**: [RUSSEL_MACRO_QUICKSTART.md](RUSSEL_MACRO_QUICKSTART.md)
- **전체 사용 가이드**: [RUSSEL_MACRO_GUIDE.md](RUSSEL_MACRO_GUIDE.md)
- **예제 코드**: [russel_macro_example.py](russel_macro_example.py)

## 🎛️ 주요 옵션

```bash
# 🚀 실시간 수강신청 모드 (권장: 미리 시작) 🆕
# ⚠️ 중요: 로컬 시간과 서버 시간 차이를 고려하여 2-3분 일찍 시작하세요!
python russel_macro.py --rapid-mode --target-time "18:58"

# 즉시 시작 (시간 대기 없이 바로 클릭) 🆕
python russel_macro.py --rapid-mode --start-immediately

# 실시간 수강신청 + 커스텀 설정 🆕
python russel_macro.py --rapid-mode --target-time "18:58:00" \
  --click-interval 0.001 --max-click-duration 180

# 입력과 클릭을 천천히 보기 (권장)
python russel_macro.py --slow-mo 1000 --typing-delay 200

# 완료 후 브라우저를 열어둠
python russel_macro.py --keep-open

# 헤드리스 모드 (백그라운드 실행)
python russel_macro.py --headless

# 빠른 실행
python russel_macro.py --fast

# 특정 강사/강의 지정
python russel_macro.py --teacher "조정식" --course "[정규/LIVE][영어] 믿어봐!"

# Playwright Inspector (디버깅)
PWDEBUG=1 python russel_macro.py

# 셸 스크립트로 실행 (대화형)
./run_russel_macro.sh
```

## 📂 파일 구조

```
russel_macro/
├── russel_macro.py                # 메인 프로그램
├── russel_macro_example.py        # 사용 예제
├── run_russel_macro.sh            # 실행 스크립트
├── RUSSEL_MACRO_GUIDE.md          # 전체 가이드
├── RUSSEL_MACRO_QUICKSTART.md     # 빠른 시작 가이드
├── requirements.txt               # 필수 패키지
├── README.md                      # 이 파일
└── screenshots/                   # 스크린샷 저장 디렉토리
    ├── step0_login_success.png
    ├── step1_initial.png
    ├── step2_english_tab.png
    ├── step3_registration.png
    └── step4_popup_*.png          # 팝업 스크린샷 🆕
```

## 🔐 주의사항

1. **로그인 정보 보안**
   - 비밀번호를 커맨드라인에 직접 입력하지 마세요
   - 환경 변수 또는 대화형 입력(`--interactive`) 사용 권장
   - Git 저장소에 로그인 정보를 커밋하지 마세요

2. **브라우저 유지**
   - `--rapid-mode` 또는 `--keep-open` 사용 시 프로그램 종료 후에도 브라우저는 계속 열려있습니다
   - 브라우저에서 결제, 약관 동의 등 추가 작업을 직접 수행할 수 있습니다
   - 작업 완료 후 브라우저 창을 직접 닫으세요

3. **윤리적 사용**
   - 개인적인 학습 및 연구 목적으로만 사용하세요
   - 과도한 요청으로 서버에 부담을 주지 마세요

4. **웹사이트 변경**
   - 웹사이트 구조가 변경되면 매크로가 작동하지 않을 수 있습니다

## 💡 문제 해결

### 타임아웃 오류

```bash
# 로그인 없이 실행
python russel_macro.py --no-login

# 브라우저를 보면서 디버깅
python russel_macro.py --interactive
```

### Playwright 설치 오류

```bash
pip uninstall playwright
pip install playwright
playwright install chromium
```

더 자세한 문제 해결 방법은 [RUSSEL_MACRO_GUIDE.md](RUSSEL_MACRO_GUIDE.md)를 참고하세요.

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

