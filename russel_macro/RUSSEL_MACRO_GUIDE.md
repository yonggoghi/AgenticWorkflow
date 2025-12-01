# 메가스터디 러셀 단과 접수 자동화 매크로 가이드

## 📋 개요

메가스터디 러셀 단과 웹사이트에서 자동으로 영어 탭을 클릭하고 특정 강의의 '접수 대기' 버튼을 클릭하는 웹 브라우저 매크로 프로그램입니다.

## 🎯 주요 기능

1. **자동 로그인**: 메가스터디 러셀 계정 자동 로그인 (환경 변수/대화형 지원)
2. **자동 사이트 방문**: 메가스터디 러셀 단과 페이지 자동 접속
3. **영어 탭 클릭**: 과목 탭에서 영어 탭 자동 선택
4. **접수 대기 버튼 클릭**: 특정 강사/강의의 접수 대기 버튼 자동 클릭
5. **🚀 실시간 수강신청 모드**: 특정 시간(예: 19:00)까지 대기 후 초고속 연속 클릭 (1ms 간격) 🆕
6. **시각적 피드백**: 클릭할 요소를 하이라이트로 표시 (빨간색/파란색 테두리)
7. **마우스 hover 효과**: 클릭 전 마우스를 요소 위로 이동
8. **천천히 타이핑**: 입력 과정을 눈으로 확인 가능
9. **스크린샷 저장**: 각 단계별 스크린샷 자동 저장
10. **유연한 설정**: 헤드리스 모드, 속도 조절 등 다양한 옵션 지원

## 🔧 설치 방법

### 1. Playwright 설치

```bash
# Playwright 패키지 설치
pip install playwright

# 브라우저 설치 (Chromium, Firefox, WebKit)
playwright install
```

### 2. 의존성 확인

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/russel_macro
pip install -r requirements.txt
```

## 🚀 사용 방법

### 로그인 방법

매크로는 세 가지 방법으로 로그인 정보를 받을 수 있습니다:

#### 1. 커맨드라인 인자

```bash
python russel_macro.py --username "아이디" --password "비밀번호"
```

#### 2. 환경 변수

```bash
export RUSSEL_USERNAME="아이디"
export RUSSEL_PASSWORD="비밀번호"
python russel_macro.py
```

#### 3. 대화형 입력 (비밀번호 숨김)

```bash
python russel_macro.py --interactive
# 아이디와 비밀번호를 입력하라는 프롬프트가 나타납니다
```

### 기본 실행

브라우저 창을 표시하면서 실행 (디버깅에 유용):

```bash
# 로그인 포함
python russel_macro.py --username "아이디" --password "비밀번호"

# 또는 환경 변수 사용
export RUSSEL_USERNAME="아이디"
export RUSSEL_PASSWORD="비밀번호"
python russel_macro.py
```

### 헤드리스 모드 (백그라운드 실행)

브라우저 창 없이 백그라운드에서 실행:

```bash
python russel_macro.py --username "아이디" --password "비밀번호" --headless
```

### 빠른 실행

느린 동작 없이 최대 속도로 실행:

```bash
python russel_macro.py --username "아이디" --password "비밀번호" --fast
```

### 특정 강사/강의 지정

원하는 강사와 강의를 지정:

```bash
python russel_macro.py --username "아이디" --password "비밀번호" \
  --teacher "조정식" --course "믿어봐! 문장·글 읽는 법을 알려줄게"
```

### 스크린샷 비활성화

스크린샷을 저장하지 않고 실행:

```bash
python russel_macro.py --username "아이디" --password "비밀번호" --no-screenshot
```

### 작업 속도 조절

각 작업 사이의 딜레이 시간 조절 (밀리초):

```bash
python russel_macro.py --username "아이디" --password "비밀번호" --slow-mo 1000  # 1초 딜레이
```

### 입력과 클릭 동작을 명확하게 보기 🆕

```bash
# 타이핑과 클릭을 천천히 보기 (권장)
python russel_macro.py --slow-mo 1000 --typing-delay 200

# 완료 후 브라우저를 열어두고 결과 확인 🆕
python russel_macro.py --keep-open

# 타이핑 효과 + 브라우저 유지 (데모/발표용 최적)
python russel_macro.py --slow-mo 1000 --typing-delay 200 --keep-open

# Playwright Inspector (최고의 디버깅 도구)
PWDEBUG=1 python russel_macro.py
```

**시각적 효과:**
- ✅ 영어 탭: 파란색 테두리 + 노란색 배경
- ✅ 접수 대기 버튼: 빨간색 테두리 + 노란색 배경
- ✅ 마우스 hover 효과로 클릭 전 요소 확인
- ✅ 타이핑 한 글자씩 보이기

### 🚀 실시간 수강신청 모드 (초고속 연속 클릭) 🆕

**목적**: 선착순 접수에서 최대한 빠르게 버튼을 클릭하여 우위를 점합니다.

**⚠️ 중요한 전략**:
- **로컬 시간과 서버 시간이 다를 수 있습니다!**
- **권장**: 정확한 시간(예: 19:00)보다 **2-3분 일찍 시작**하세요 (예: 18:58)
- 미리 시작하면 서버 시간 차이와 네트워크 지연을 보완할 수 있습니다

**동작 원리**:
1. 로그인 → 사이트 방문 → 영어 탭 클릭 → 버튼 찾기
2. 목표 시간까지 대기 (또는 즉시 시작)
3. 1ms 간격으로 버튼을 초고속 연속 클릭
4. 페이지 변화(팝업/URL 변경/성공 메시지) 감지 시 즉시 중단하고 브라우저 유지

**사용법**:

```bash
# ⭐ 권장: 2-3분 일찍 시작 (18:58부터 클릭, 19:00 접수 시작 대비)
export RUSSEL_USERNAME="아이디"
export RUSSEL_PASSWORD="비밀번호"
python russel_macro.py --rapid-mode --target-time "18:58"

# ⚡ 즉시 시작 (시간 대기 없이 바로 클릭)
python russel_macro.py --rapid-mode --start-immediately

# 기본 실시간 수강신청 (19:00부터 클릭 - 늦을 수 있음)
python russel_macro.py --rapid-mode --target-time "19:00"

# 초 단위까지 지정 (18:58:30)
python russel_macro.py --rapid-mode --target-time "18:58:30"

# 클릭 간격 조정 (0.5ms로 더 빠르게)
python russel_macro.py --rapid-mode --click-interval 0.0005

# 최대 클릭 시간 조정 (180초 = 3분, 접수 시작 시간이 불확실할 때)
python russel_macro.py --rapid-mode --target-time "18:58" --max-click-duration 180

# 대화형 입력 + 실시간 수강신청
python russel_macro.py --rapid-mode --interactive --target-time "18:58"
```

**주의사항**:
- ⚠️ **핵심**: 로컬 시간과 서버 시간 차이를 고려하여 **2-3분 일찍 시작**하세요!
- ⚠️ 실시간 수강신청 모드는 자동으로 `--keep-open` 옵션이 활성화됩니다.
- ⚠️ 미리 테스트 실행하여 버튼이 정확히 찾아지는지 확인하세요.
- ⚠️ `--max-click-duration`을 충분히 길게 설정하면 (예: 180초) 접수 시작 시간이 불확실해도 안전합니다.
- ⚠️ 클릭 간격이 너무 짧으면 브라우저나 서버에서 차단할 수 있습니다.

**브라우저 유지 및 수동 작업**:
- ✅ 프로그램 종료 후에도 **브라우저는 계속 열려있습니다**
- ✅ 접수 후 나타나는 팝업/페이지에서 **직접 작업 가능**:
  - 결제 정보 입력
  - 약관 동의 체크
  - 추가 정보 입력
  - 결제 버튼 클릭
- ✅ 작업 완료 후 **브라우저 창을 직접 닫으세요**
- ✅ 프로그램을 종료하려면 터미널에서 Enter 키를 누르면 됩니다 (브라우저는 계속 열림)

### 로그인 없이 실행

로그인이 필요 없는 경우 (예: 이미 로그인되어 있는 경우):

```bash
python russel_macro.py --no-login
```

## 📊 출력 결과

### 실행 로그

매크로 실행 중 다음과 같은 로그가 출력됩니다:

```
[2025-11-17 10:30:00] [INFO] ================================================================================
[2025-11-17 10:30:00] [INFO] 메가스터디 러셀 단과 접수 자동화 시작
[2025-11-17 10:30:00] [INFO] ================================================================================
[2025-11-17 10:30:00] [INFO] 로그인 단계 시작
[2025-11-17 10:30:00] [INFO] 로그인 페이지로 이동 중...
[2025-11-17 10:30:02] [INFO] 로그인 버튼 찾는 중...
[2025-11-17 10:30:03] [INFO] 아이디/비밀번호 입력 중...
[2025-11-17 10:30:04] [INFO] 로그인 버튼 클릭 중...
[2025-11-17 10:30:05] [INFO] 로그인 성공!
[2025-11-17 10:30:05] [INFO] 사이트 방문 중: https://russelbd.megastudy.net/...
[2025-11-17 10:30:07] [INFO] 사이트 방문 완료
[2025-11-17 10:30:07] [INFO] 영어 탭 찾는 중...
[2025-11-17 10:30:08] [INFO] 영어 탭 클릭 완료
[2025-11-17 10:30:10] [INFO] 접수 대기 버튼 찾는 중...
[2025-11-17 10:30:11] [INFO] 접수 대기 버튼 클릭 완료!
[2025-11-17 10:30:12] [INFO] 팝업 창 감지: https://russelbd.megastudy.net/...
[2025-11-17 10:30:13] [INFO] 팝업 스크린샷 저장: screenshots/step4_popup_20251117_103013.png
[2025-11-17 10:30:13] [INFO] 팝업 제목: 접수 대기 안내
[2025-11-17 10:30:13] [INFO] 팝업이 열렸습니다. 5초 후 자동으로 계속 진행합니다...
[2025-11-17 10:30:18] [INFO] ================================================================================
[2025-11-17 10:30:18] [INFO] 매크로 실행 완료!
[2025-11-17 10:30:18] [INFO] ================================================================================
```

### 스크린샷 파일

각 단계별 스크린샷이 `screenshots/` 디렉토리에 저장됩니다:

- `step0_login_success.png`: 로그인 성공 후
- `step1_initial.png`: 초기 페이지 로드 후
- `step2_english_tab.png`: 영어 탭 클릭 후
- `step3_registration.png`: 접수 대기 버튼 클릭 후
- `step4_popup_YYYYMMDD_HHMMSS.png`: 팝업 창 스크린샷 (새 창으로 열리는 경우) 🆕
- `step4_modal_YYYYMMDD_HHMMSS.png`: 페이지 내 모달 스크린샷 (팝업이 아닌 경우) 🆕

오류 발생 시:
- `step0_login_error.png`: 로그인 실패 시
- `step2_error.png`: 영어 탭 클릭 실패 시
- `step3_error.png`: 접수 대기 버튼 클릭 실패 시

## 🎛️ 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--username`, `-u` | 로그인 아이디 | 환경 변수 `RUSSEL_USERNAME` |
| `--password`, `-p` | 로그인 비밀번호 | 환경 변수 `RUSSEL_PASSWORD` |
| `--interactive`, `-i` | 대화형 로그인 (비밀번호 숨김) | False |
| `--no-login` | 로그인 단계 건너뛰기 | False |
| `--headless` | 헤드리스 모드 (브라우저 창 숨김) | False |
| `--fast` | 빠른 실행 (slow_mo=0) | False |
| `--teacher` | 강사 이름 | "조정식" |
| `--course` | 강의명 키워드 | "[정규/LIVE][영어] 믿어봐! 문장·글 읽는 법을 알려줄게 (오전반)" |
| `--no-screenshot` | 스크린샷 저장 안 함 | False |
| `--slow-mo` | 작업 속도 조절 (밀리초) | 500 |
| `--typing-delay` | 타이핑 속도 조절 (밀리초/글자) | 100 |
| `--keep-open` | 완료 후 브라우저를 열어둠 | False |
| `--rapid-mode` | 🚀 실시간 수강신청 모드 (초고속 연속 클릭) 🆕 | False |
| `--target-time` | 실시간 수강신청 목표 시간 (HH:MM 또는 HH:MM:SS) 🆕 | 19:00 |
| `--start-immediately` | ⚡ 시간 대기 없이 즉시 클릭 시작 🆕 | False |
| `--click-interval` | 연속 클릭 간격 (초) 🆕 | 0.001 (1ms) |
| `--max-click-duration` | 최대 클릭 시도 시간 (초) 🆕 | 30 |

## 🔍 문제 해결

### Playwright 설치 오류

```bash
# Playwright 재설치
pip uninstall playwright
pip install playwright
playwright install chromium
```

### 브라우저 실행 오류

macOS에서 권한 문제가 발생할 수 있습니다:

```bash
# 시스템 보안 설정에서 브라우저 실행 권한 허용
# 시스템 환경설정 > 보안 및 개인정보 보호 > 일반
```

### 요소를 찾을 수 없음

웹사이트 구조가 변경되었을 수 있습니다. 디버깅 모드로 실행:

```bash
# 느린 속도로 실행하여 각 단계 확인
python russel_macro.py --slow-mo 2000
```

### 타임아웃 오류

로그인 페이지 접속 시 타임아웃이 발생하는 경우:

```
[ERROR] 로그인 실패: Page.goto: Timeout 30000ms exceeded.
```

**해결 방법:**

1. **네트워크 확인**: 인터넷 연결 상태를 확인하세요

2. **타임아웃 증가**: 최신 버전은 이미 60초로 설정되어 있습니다

3. **로그인 건너뛰기**: 로그인 없이 실행 후 수동으로 로그인
   ```bash
   python russel_macro.py --no-login
   # 브라우저에서 수동으로 로그인 후 계속 진행
   ```

4. **헤드리스 모드 비활성화**: 브라우저를 보면서 디버깅
   ```bash
   python russel_macro.py --interactive  # 헤드리스 모드 OFF
   ```

5. **네트워크 환경**: VPN이나 프록시를 사용 중이면 비활성화해보세요

## 📝 Python 코드에서 사용

매크로를 Python 스크립트에서 직접 사용:

```python
from russel_macro import RusselMacro
import os

# 환경 변수에서 로그인 정보 가져오기
username = os.getenv('RUSSEL_USERNAME')
password = os.getenv('RUSSEL_PASSWORD')

# 컨텍스트 매니저 사용
with RusselMacro(headless=False, slow_mo=500) as macro:
    success = macro.run(
        username=username,
        password=password,
        teacher_name="조정식",
        course_name="믿어봐! 문장·글 읽는 법을 알려줄게",
        screenshot=True
    )
    
    if success:
        print("매크로 실행 성공!")
    else:
        print("매크로 실행 실패!")
```

### 단계별 제어

```python
from russel_macro import RusselMacro
import getpass

# 대화형 로그인 정보 입력
username = input("아이디: ")
password = getpass.getpass("비밀번호: ")

with RusselMacro(headless=False, slow_mo=1000) as macro:
    # 0단계: 로그인
    if not macro.login(username, password):
        print("로그인 실패")
        exit(1)
    
    # 1단계: 사이트 방문
    if not macro.visit_site():
        print("사이트 방문 실패")
        exit(1)
    
    # 2단계: 영어 탭 클릭
    if not macro.click_english_tab():
        print("영어 탭 클릭 실패")
        exit(1)
    
    # 3단계: 접수 대기 버튼 클릭
    if not macro.click_registration_button("조정식", "믿어봐"):
        print("접수 대기 버튼 클릭 실패")
        exit(1)
    
    print("전체 프로세스 완료!")
```

## 🔐 주의사항

1. **로그인 정보 보안**: 
   - 비밀번호를 커맨드라인에 직접 입력하지 마세요 (셸 히스토리에 남습니다)
   - 환경 변수 또는 대화형 입력(`--interactive`) 사용을 권장합니다
   - `.bashrc`나 `.zshrc`에 비밀번호를 평문으로 저장하지 마세요
   - Git 저장소에 로그인 정보가 포함된 파일을 커밋하지 마세요

2. **윤리적 사용**: 이 매크로는 개인적인 학습 및 연구 목적으로만 사용하세요.

3. **서버 부하**: 과도한 요청은 서버에 부담을 줄 수 있으니 적절히 사용하세요.

4. **웹사이트 변경**: 웹사이트 구조가 변경되면 매크로가 작동하지 않을 수 있습니다.

5. **자동화 정책**: 웹사이트의 자동화 정책을 확인하고 준수하세요.

## 📚 관련 문서

- [Playwright Documentation](https://playwright.dev/python/)
- [Web Crawler Guide](WEB_CRAWLER_GUIDE.md)
- [Product Crawler Guide](PRODUCT_CRAWLER_GUIDE.md)

## 🐛 버그 리포트

문제가 발생하면 다음 정보와 함께 리포트해주세요:

1. 실행 명령어
2. 오류 메시지
3. 스크린샷 (가능한 경우)
4. 환경 정보 (OS, Python 버전)

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로만 사용됩니다.

