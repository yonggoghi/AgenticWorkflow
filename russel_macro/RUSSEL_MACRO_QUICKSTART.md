# 러셀 단과 접수 매크로 빠른 시작

## 5분 안에 시작하기 🚀

### 1단계: Playwright 설치 (최초 1회만)

```bash
# Playwright 설치
pip install playwright

# 브라우저 설치 (Chromium)
playwright install chromium
```

### 2단계: 로그인 정보 설정

3가지 방법 중 하나를 선택하세요:

#### 방법 1: 환경 변수 (권장)

```bash
export RUSSEL_USERNAME="your_id"
export RUSSEL_PASSWORD="your_password"
```

#### 방법 2: 대화형 입력

매크로 실행 시 `--interactive` 옵션 사용

#### 방법 3: 커맨드라인 인자

실행 시 `--username`과 `--password` 옵션 사용 (보안상 비추천)

### 3단계: 매크로 실행

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/russel_macro

# 방법 1: 환경 변수 사용
python russel_macro.py

# 방법 2: 대화형 입력
python russel_macro.py --interactive

# 방법 3: 커맨드라인 인자 (보안상 비추천)
python russel_macro.py --username "아이디" --password "비밀번호"
```

끝입니다! 브라우저가 자동으로 열리고 다음 작업을 수행합니다:
1. ✅ 자동 로그인
2. ✅ 러셀 단과 사이트 방문
3. ✅ 영어 탭 클릭
4. ✅ 조정식 강사의 "믿어봐! 문장·글 읽는 법을 알려줄게" 강의 접수 대기 버튼 클릭

## 🚀 실시간 수강신청 모드 (초고속 클릭) 🆕

**⚠️ 핵심 전략**: 로컬 시간과 서버 시간이 다를 수 있으므로 **2-3분 일찍 시작**하세요!

### 권장 방법 (미리 시작)

```bash
# 1단계: 환경 변수 설정
export RUSSEL_USERNAME="아이디"
export RUSSEL_PASSWORD="비밀번호"

# 2단계: 2-3분 일찍 시작 (19:00 접수 시작 대비 → 18:58부터 클릭)
python russel_macro.py --rapid-mode --target-time "18:58"

# 또는 즉시 시작 (프로그램 실행 즉시 클릭 시작)
python russel_macro.py --rapid-mode --start-immediately
```

### 다른 시간 설정

```bash
# 19:30 접수 시작 → 18:28부터 클릭
python russel_macro.py --rapid-mode --target-time "19:28"

# 20:00 접수 시작 → 19:58부터 클릭
python russel_macro.py --rapid-mode --target-time "19:58"

# 최대 클릭 시간을 3분으로 설정 (접수 시간이 불확실할 때)
python russel_macro.py --rapid-mode --target-time "18:58" --max-click-duration 180
```

**동작**:
- 로그인 → 사이트 방문 → 영어 탭 클릭 → 버튼 찾기
- 목표 시간까지 대기 (또는 즉시 시작)
- 1ms 간격으로 버튼을 초고속 연속 클릭
- 팝업이나 페이지 변화 감지 시 자동 중단 및 브라우저 유지

**왜 미리 시작해야 하나요?**
- 내 컴퓨터 시간과 서버 시간이 다를 수 있음
- 네트워크 지연 시간 보정
- "접수 대기" 상태일 때도 계속 클릭하다가 "접수 가능" 상태로 바뀌는 순간 바로 접수

**접수 후 작업**:
- ✅ **프로그램이 종료되어도 브라우저는 계속 열려있습니다**
- ✅ 접수 후 나타나는 팝업/페이지에서 **직접 작업하세요**:
  - 결제 정보 입력
  - 약관 동의
  - 추가 정보 입력
  - 결제 완료
- ✅ 모든 작업 완료 후 브라우저 창을 직접 닫으세요

## 다른 강의 선택하기

```bash
# 환경 변수 사용 (권장)
export RUSSEL_USERNAME="아이디"
export RUSSEL_PASSWORD="비밀번호"
python russel_macro.py --teacher "강사명" --course "강의명 일부"

# 또는 대화형 입력
python russel_macro.py --interactive --teacher "강사명" --course "강의명 일부"
```

## 백그라운드 실행 (브라우저 창 숨김)

```bash
# 환경 변수 사용
python russel_macro.py --headless

# 또는 대화형 입력
python russel_macro.py --interactive --headless
```

## 작업 확인하기 (브라우저 유지) 🆕

매크로 완료 후 브라우저를 열어두고 결과를 확인:

```bash
# 브라우저를 닫지 않고 유지
python russel_macro.py --keep-open

# 천천히 실행 + 브라우저 유지 (권장)
python russel_macro.py --slow-mo 1000 --typing-delay 200 --keep-open
```

완료 후 브라우저가 열린 상태로 유지되며, Enter 키를 눌러 종료할 수 있습니다.

## 대화형 실행 (셸 스크립트)

```bash
# 셸 스크립트로 실행하면 메뉴가 나타납니다
./run_russel_macro.sh
```

실행 모드 선택:
- 1: 기본 실행 (브라우저 표시)
- 2: 헤드리스 모드
- 3: 빠른 실행
- 4: 예제 실행

## 결과 확인

실행이 완료되면 `screenshots/` 디렉토리에 스크린샷이 저장됩니다:

- `step0_login_success.png`: 로그인 성공 후
- `step1_initial.png`: 초기 페이지
- `step2_english_tab.png`: 영어 탭 클릭 후
- `step3_registration.png`: 접수 대기 버튼 클릭 후
- `step4_popup_*.png` 또는 `step4_modal_*.png`: 팝업/모달 스크린샷 🆕

## 문제 해결

### Playwright 설치 오류

```bash
pip uninstall playwright
pip install playwright
playwright install chromium
```

### 타임아웃 오류 (로그인 페이지 접속 실패)

네트워크 문제나 사이트 응답 지연:

```bash
# 로그인 건너뛰고 실행
python russel_macro.py --no-login

# 또는 브라우저를 보면서 실행 (헤드리스 모드 OFF)
python russel_macro.py --interactive
```

### "접수 대기 버튼을 찾을 수 없습니다" 오류

웹사이트 구조가 변경되었을 수 있습니다. 천천히 실행하여 확인:

```bash
# 환경 변수 설정 후
export RUSSEL_USERNAME="아이디"
export RUSSEL_PASSWORD="비밀번호"
python russel_macro.py --slow-mo 2000
```

## 추가 정보

자세한 사용법은 [RUSSEL_MACRO_GUIDE.md](RUSSEL_MACRO_GUIDE.md)를 참고하세요.

## 주의사항

⚠️ **로그인 정보 보안**: 비밀번호를 커맨드라인에 직접 입력하지 마세요 (히스토리에 남습니다)  
⚠️ **환경 변수 사용 권장**: `export RUSSEL_USERNAME="아이디"` `RUSSEL_PASSWORD="비밀번호"`  
⚠️ **대화형 입력 권장**: `--interactive` 옵션 사용 시 비밀번호가 화면에 표시되지 않습니다  
⚠️ **교육 및 연구 목적**: 이 매크로는 개인적인 학습 및 연구 목적으로만 사용하세요  
⚠️ **서버 부담 주의**: 과도한 요청은 서버에 부담을 줄 수 있으니 적절히 사용하세요

