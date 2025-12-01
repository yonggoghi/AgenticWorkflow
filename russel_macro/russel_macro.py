#!/usr/bin/env python3
"""
메가스터디 러셀 단과 접수 자동화 매크로
- 로그인 자동화
- 사이트 방문 및 국어 탭 클릭
- 특정 강의의 '결제하기' 버튼 클릭
"""

import argparse
import os
import time
import getpass
from datetime import datetime, timedelta
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Error: Playwright is not installed.")
    print("Install with: pip install playwright && playwright install")
    exit(1)


class RusselMacro:
    """메가스터디 러셀 단과 접수 자동화 클래스"""
    
    def __init__(self, headless: bool = False, slow_mo: int = 500, typing_delay: int = 100, keep_open: bool = False):
        """
        Args:
            headless: 헤드리스 모드 (False면 브라우저 UI 표시)
            slow_mo: 작업 속도 조절 (밀리초, 디버깅용)
            typing_delay: 타이핑 속도 조절 (밀리초, 한 글자당)
            keep_open: 완료 후 브라우저를 열어둘지 여부
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.typing_delay = typing_delay
        self.keep_open = keep_open
        self.base_url = "https://russelbd.megastudy.net/russel/campus_common/russel_danka/russel_danka_new.asp?idx=2201"
        self.browser = None
        self.page = None
        self.context = None
        
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        self.context = self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        self.page = self.context.new_page()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        if self.keep_open:
            self.log("="*80)
            self.log("✅ 작업 완료! 브라우저는 계속 열려있습니다.")
            self.log("="*80)
            self.log("📌 브라우저에서 필요한 작업을 계속하세요:")
            self.log("   - 결제 정보 입력")
            self.log("   - 약관 동의")
            self.log("   - 추가 정보 확인")
            self.log("   - 기타 필요한 작업")
            self.log("")
            self.log("⚠️  프로그램 종료 시 브라우저도 함께 닫힙니다.")
            self.log("⚠️  작업을 모두 완료한 후 Enter를 누르세요.")
            self.log("="*80)
            
            # 브라우저를 유지하면서 대기
            try:
                input("\n작업 완료 후 Enter를 누르면 브라우저가 닫힙니다...")
            except (KeyboardInterrupt, EOFError):
                pass
            
            self.log("\n프로그램을 종료합니다...")
        
        # 브라우저 닫기
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def log(self, message: str, level: str = "INFO"):
        """로그 출력"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
    
    def login(self, username: str, password: str):
        """로그인 수행"""
        self.log("로그인 페이지로 이동 중...")
        
        try:
            # 로그인 페이지로 직접 이동
            login_url = "https://russelbd.megastudy.net/russel/member/login.asp"
            self.page.goto(login_url, wait_until='domcontentloaded', timeout=60000)
            self.page.wait_for_timeout(2000)  # 페이지 안정화 대기
            
            self.log("로그인 페이지 로드 완료")
            self.log("아이디/비밀번호 입력 중...")
            
            # 아이디 입력 필드 찾기
            id_selectors = [
                "input[name='user_id']",
                "input[name='userid']",
                "input[name='id']",
                "input[id='user_id']",
                "input[id='userid']",
                "input[type='text']",
            ]
            
            id_field = None
            for selector in id_selectors:
                try:
                    element = self.page.wait_for_selector(selector, timeout=3000)
                    if element and element.is_visible():
                        self.log(f"아이디 입력 필드 발견: {selector}")
                        id_field = element
                        break
                except:
                    continue
            
            if not id_field:
                self.log("아이디 입력 필드를 찾을 수 없습니다.", "ERROR")
                return False
            
            # 비밀번호 입력 필드 찾기
            pw_selectors = [
                "input[name='user_pwd']",
                "input[name='password']",
                "input[name='pwd']",
                "input[id='user_pwd']",
                "input[id='password']",
                "input[type='password']",
            ]
            
            pw_field = None
            for selector in pw_selectors:
                try:
                    element = self.page.wait_for_selector(selector, timeout=3000)
                    if element and element.is_visible():
                        self.log(f"비밀번호 입력 필드 발견: {selector}")
                        pw_field = element
                        break
                except:
                    continue
            
            if not pw_field:
                self.log("비밀번호 입력 필드를 찾을 수 없습니다.", "ERROR")
                return False
            
            # 아이디/비밀번호 입력
            self.log(f"아이디 입력: {username[:2]}*** (타이핑 속도: {self.typing_delay}ms/글자)")
            # type()을 사용하면 천천히 타이핑하는 것처럼 보입니다
            id_field.click()  # 필드 클릭
            id_field.fill('')  # 기존 값 지우기
            id_field.type(username, delay=self.typing_delay)  # 타이핑 딜레이
            self.page.wait_for_timeout(500)
            
            self.log("비밀번호 입력: ***")
            pw_field.click()  # 필드 클릭
            pw_field.fill('')  # 기존 값 지우기
            pw_field.type(password, delay=self.typing_delay)  # 타이핑 딜레이
            self.page.wait_for_timeout(500)
            
            self.log("로그인 시도 중...")
            
            # 방법 1: Enter 키로 제출 (가장 확실함)
            pw_field.press("Enter")
            self.log("Enter 키로 로그인 제출")
            
            # 또는 방법 2: 로그인 버튼 클릭도 시도
            # submit_selectors = [
            #     "button[type='submit']",
            #     "input[type='submit']",
            #     "button:has-text('로그인')",
            #     "input[value='로그인']",
            #     "a:has-text('로그인')",
            # ]
            # 
            # for selector in submit_selectors:
            #     try:
            #         element = self.page.query_selector(selector)
            #         if element and element.is_visible():
            #             self.log(f"로그인 버튼도 클릭: {selector}")
            #             element.click()
            #             break
            #     except:
            #         continue
            
            # 로그인 완료 대기
            self.page.wait_for_timeout(3000)
            
            # 로그인 성공 확인 (여러 방법 시도)
            current_url = self.page.url
            self.log(f"현재 URL: {current_url}")
            
            # 1. URL 변경 확인 (로그인 페이지에서 다른 페이지로 이동했는지)
            if 'login.asp' not in current_url:
                self.log("로그인 성공! (URL 변경 확인)")
                return True
            
            # 2. 페이지 텍스트로 확인
            page_text = self.page.inner_text('body')
            if '로그아웃' in page_text or 'logout' in page_text.lower():
                self.log("로그인 성공! (로그아웃 버튼 확인)")
                return True
            
            # 3. 로그인 오류 메시지 확인
            error_messages = [
                '아이디', '비밀번호', '확인', '일치하지', '존재하지',
                '입력', '다시', '실패'
            ]
            has_error = any(msg in page_text for msg in error_messages)
            
            if has_error and 'login.asp' in current_url:
                self.log("로그인 실패: 아이디 또는 비밀번호를 확인하세요.", "ERROR")
                self.log(f"페이지 내용 일부: {page_text[:200]}", "DEBUG")
                return False
            
            # 4. 불확실한 경우 성공으로 간주하고 계속 진행
            self.log("로그인 상태를 명확히 확인할 수 없습니다. 계속 진행합니다.", "WARNING")
            return True
                
        except Exception as e:
            self.log(f"로그인 실패: {e}", "ERROR")
            return False
    
    def visit_site(self):
        """1단계: 사이트 방문"""
        self.log(f"사이트 방문 중: {self.base_url}")
        try:
            # domcontentloaded는 networkidle보다 빠르고 안정적
            self.page.goto(self.base_url, wait_until='domcontentloaded', timeout=60000)
            self.page.wait_for_timeout(3000)  # 페이지 로딩 대기
            self.log("사이트 방문 완료")
            return True
        except Exception as e:
            self.log(f"사이트 방문 실패: {e}", "ERROR")
            return False
    
    def click_korean_tab(self):
        """2단계: 국어 탭 클릭"""
        self.log("국어 탭 찾는 중...")
        
        try:
            # 여러 선택자 시도
            selectors = [
                "text='국어'",  # 정확한 텍스트 매칭
                "a:has-text('국어')",  # 링크 태그 내 텍스트
                "//a[contains(text(), '국어')]",  # XPath
                "li:has-text('국어') a",  # 리스트 아이템 내 링크
            ]
            
            clicked = False
            for selector in selectors:
                try:
                    self.log(f"선택자 시도: {selector}")
                    element = self.page.wait_for_selector(selector, timeout=5000)
                    if element:
                        # 요소가 보일 때까지 대기
                        self.log("국어 탭으로 스크롤 중...")
                        element.scroll_into_view_if_needed()
                        self.page.wait_for_timeout(800)
                        
                        # 마우스 hover
                        self.log("국어 탭에 마우스 올리기...")
                        element.hover()
                        self.page.wait_for_timeout(800)
                        
                        # 하이라이트 효과
                        try:
                            self.page.evaluate("""
                                (el) => {
                                    el.style.outline = '3px solid blue';
                                    el.style.backgroundColor = '#ffffcc';
                                }
                            """, element)
                            self.page.wait_for_timeout(1000)
                        except:
                            pass
                        
                        # 클릭
                        self.log("국어 탭 클릭 중...")
                        element.click()
                        self.log("국어 탭 클릭 완료")
                        self.page.wait_for_timeout(2000)  # 탭 전환 대기
                        clicked = True
                        break
                except PlaywrightTimeoutError:
                    continue
                except Exception as e:
                    self.log(f"선택자 {selector} 실패: {e}", "WARNING")
                    continue
            
            if not clicked:
                self.log("국어 탭을 찾을 수 없습니다. 페이지 구조를 확인합니다.", "WARNING")
                # 디버깅: 페이지의 모든 텍스트 출력
                page_text = self.page.inner_text('body')
                if '국어' in page_text:
                    self.log("페이지에 '국어' 텍스트가 존재합니다.", "INFO")
                else:
                    self.log("페이지에 '국어' 텍스트가 없습니다.", "ERROR")
                return False
                
            return True
            
        except Exception as e:
            self.log(f"국어 탭 클릭 실패: {e}", "ERROR")
            return False
    
    def click_registration_button(self, teacher_name: str = "강민철", 
                                  course_name: str = "[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)"):
        """3단계: 결제하기 버튼 클릭"""
        self.log(f"결제하기 버튼 찾는 중... (강사: {teacher_name}, 강의: {course_name})")
        
        try:
            # 방법 1: 텍스트로 직접 찾기
            button_selectors = [
                "text='결제하기'",
                "button:has-text('결제하기')",
                "a:has-text('결제하기')",
                "input[value='결제하기']",
                "//*[contains(text(), '결제하기')]",
            ]
            
            # 모든 '결제하기' 버튼 찾기
            self.log("페이지에서 모든 '결제하기' 버튼 검색 중...")
            all_buttons = self.page.query_selector_all("text='결제하기'")
            self.log(f"총 {len(all_buttons)}개의 '결제하기' 버튼 발견")
            
            if len(all_buttons) == 0:
                self.log("결제하기 버튼을 찾을 수 없습니다.", "ERROR")
                return False
            
            # 특정 강의의 버튼을 찾기 위해 상위 요소 확인
            target_button = None
            best_match_score = 0
            best_match_button = None
            
            for idx, button in enumerate(all_buttons):
                # 버튼의 부모 요소에서 강사명이나 강의명 확인
                parent = button.evaluate_handle('el => el.closest("tr, div, li")')
                if parent:
                    parent_text = parent.evaluate('el => el.innerText')
                    self.log(f"버튼 {idx+1} 주변 텍스트: {parent_text[:150]}...")
                    
                    # 매칭 점수 계산
                    match_score = 0
                    
                    # 1. 강사명 확인 (필수)
                    if teacher_name in parent_text:
                        match_score += 1
                        
                        # 2. 강의명 전체 문자열 매칭 (가장 정확)
                        if course_name in parent_text:
                            match_score += 10
                            self.log(f"✅ 버튼 {idx+1}: 강의명 정확히 일치! (점수: {match_score})")
                        else:
                            # 3. 핵심 키워드 매칭
                            # "[정규/LIVE]" 또는 "[정규]" 구분
                            if "[정규/LIVE]" in course_name and "[정규/LIVE]" in parent_text:
                                match_score += 5
                                self.log(f"✅ 버튼 {idx+1}: [정규/LIVE] 매칭 (점수: {match_score})")
                            elif "[정규]" in course_name and "[정규]" in parent_text and "[정규/LIVE]" not in parent_text:
                                match_score += 5
                                self.log(f"✅ 버튼 {idx+1}: [정규] 매칭 (점수: {match_score})")
                            
                            # 4. "오전반", "오후반" 등 시간대 매칭
                            for time_keyword in ["(오전반)", "(오후반)", "(종일반)"]:
                                if time_keyword in course_name and time_keyword in parent_text:
                                    match_score += 2
                                    break
                        
                        # 더 높은 점수면 업데이트
                        if match_score > best_match_score:
                            best_match_score = match_score
                            best_match_button = button
                            self.log(f"🎯 현재 최고 매칭: 버튼 {idx+1} (점수: {match_score})")
            
            # 최고 매칭 버튼 사용
            if best_match_button:
                target_button = best_match_button
                self.log(f"✅ 최종 선택: 매칭 점수 {best_match_score}점인 버튼")
            else:
                self.log(f"특정 강의를 찾지 못했습니다. 첫 번째 '결제하기' 버튼을 사용합니다.", "WARNING")
                target_button = all_buttons[0]
            
            # 버튼 클릭 전 준비
            self.log("결제하기 버튼으로 이동 중...")
            
            # 버튼이 화면에 보이도록 스크롤
            target_button.scroll_into_view_if_needed()
            self.page.wait_for_timeout(1000)  # 스크롤 후 대기
            
            # 마우스를 버튼으로 이동 (hover 효과)
            self.log("결제하기 버튼에 마우스 올리기...")
            target_button.hover()
            self.page.wait_for_timeout(1000)  # hover 효과 확인
            
            # 버튼 하이라이트 (시각적 효과)
            try:
                self.page.evaluate("""
                    (element) => {
                        element.style.outline = '3px solid red';
                        element.style.backgroundColor = 'yellow';
                    }
                """, target_button)
                self.log("결제하기 버튼 하이라이트 완료")
                self.page.wait_for_timeout(1500)  # 하이라이트 확인
            except:
                pass
            
            # 팝업(새 창) 대기 설정
            self.log("결제하기 버튼 클릭 중...")
            try:
                # 새 창/팝업 이벤트 대기
                with self.context.expect_page() as popup_info:
                    target_button.click()
                    self.log("결제하기 버튼 클릭 완료!")
                
                # 팝업 처리
                popup = popup_info.value
                self.log(f"팝업 창 감지: {popup.url}")
                
                # 팝업 로딩 대기
                popup.wait_for_load_state('domcontentloaded', timeout=10000)
                self.page.wait_for_timeout(2000)
                
                # 팝업 스크린샷 저장
                screenshot_dir = Path('screenshots')
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                popup_screenshot = screenshot_dir / f"step4_popup_{timestamp}.png"
                popup.screenshot(path=str(popup_screenshot), full_page=True)
                self.log(f"팝업 스크린샷 저장: {popup_screenshot}")
                
                # 팝업 제목과 내용 로그
                popup_title = popup.title()
                self.log(f"팝업 제목: {popup_title}")
                
                # 팝업 내용 일부 출력
                popup_text = popup.inner_text('body')
                self.log(f"팝업 내용 일부: {popup_text[:200]}")
                
                # 팝업 유지 (사용자가 확인할 수 있도록)
                self.log("팝업이 열렸습니다. 5초 후 자동으로 계속 진행합니다...")
                popup.wait_for_timeout(5000)
                
            except Exception as popup_error:
                self.log(f"팝업이 없거나 처리 중 오류: {popup_error}", "WARNING")
                self.log("팝업 대신 페이지 내 모달일 수 있습니다. 페이지 스크린샷을 저장합니다.")
                
                # 팝업이 없는 경우 현재 페이지의 스크린샷 저장
                self.page.wait_for_timeout(2000)
                screenshot_dir = Path('screenshots')
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                modal_screenshot = screenshot_dir / f"step4_modal_{timestamp}.png"
                self.page.screenshot(path=str(modal_screenshot), full_page=True)
                self.log(f"페이지/모달 스크린샷 저장: {modal_screenshot}")
                
                # 추가 대기
                self.page.wait_for_timeout(3000)
            
            return True
            
        except Exception as e:
            self.log(f"결제하기 버튼 클릭 실패: {e}", "ERROR")
            return False
    
    def save_screenshot(self, filename: str = None):
        """스크린샷 저장"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"russel_macro_{timestamp}.png"
        
        screenshot_dir = Path('screenshots')
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshot_dir / filename
        
        self.page.screenshot(path=str(screenshot_path), full_page=True)
        self.log(f"스크린샷 저장: {screenshot_path}")
        return screenshot_path
    
    def wait_until_time(self, target_time_str: str, allow_past: bool = True):
        """특정 시간까지 대기
        
        Args:
            target_time_str: 목표 시간 문자열 (예: "19:00", "19:00:00")
            allow_past: True면 이미 지난 시간일 경우 즉시 시작, False면 내일로 설정
        """
        # 목표 시간 파싱
        time_parts = target_time_str.split(':')
        target_hour = int(time_parts[0])
        target_minute = int(time_parts[1]) if len(time_parts) > 1 else 0
        target_second = int(time_parts[2]) if len(time_parts) > 2 else 0
        
        now = datetime.now()
        target = now.replace(hour=target_hour, minute=target_minute, second=target_second, microsecond=0)
        
        # 목표 시간이 이미 지났을 때 처리
        if target <= now:
            if allow_past:
                self.log(f"⚠️  목표 시간({target_time_str})이 이미 지났습니다. 즉시 시작합니다.", "WARNING")
                self.log(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                return
            else:
                target += timedelta(days=1)
        
        wait_seconds = (target - now).total_seconds()
        
        self.log(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"목표 시간: {target.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"대기 시간: {wait_seconds:.1f}초 ({wait_seconds/60:.1f}분)")
        
        # 남은 시간이 60초 이상이면 진행 상황 표시
        if wait_seconds > 60:
            self.log("대기 중... (1분마다 로그 출력)")
            while True:
                now = datetime.now()
                remaining = (target - now).total_seconds()
                
                if remaining <= 60:
                    break
                
                self.log(f"남은 시간: {remaining:.0f}초 ({remaining/60:.1f}분)")
                time.sleep(60)
        
        # 마지막 60초는 더 자주 체크
        if wait_seconds > 0:
            remaining = (target - datetime.now()).total_seconds()
            if remaining > 10:
                self.log(f"최종 대기: {remaining:.1f}초")
                time.sleep(max(0, remaining - 5))  # 5초 전까지 대기
            
            # 마지막 5초는 정밀하게 대기
            while datetime.now() < target:
                time.sleep(0.001)  # 1ms 단위로 체크
        
        self.log(f"⏰ 클릭 시작! {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    def rapid_click_korean_tab_until_payment_button(self, teacher_name: str, course_name: str,
                                                     max_duration: int = 300, click_interval: float = 0.001):
        """국어 탭을 계속 클릭하면서 결제하기 버튼이 나타날 때까지 대기
        
        Args:
            teacher_name: 강사 이름
            course_name: 강의명
            max_duration: 최대 클릭 시도 시간 (초)
            click_interval: 클릭 간격 (초)
        
        Returns:
            button_element or None: 찾은 결제하기 버튼 요소, 실패 시 None
        """
        self.log("="*80)
        self.log("🔄 국어 탭을 계속 클릭하면서 '결제하기' 버튼 출현 대기 중...")
        self.log(f"설정: 클릭 간격 {click_interval*1000:.1f}ms, 최대 {max_duration}초")
        self.log("="*80)
        
        start_time = time.time()
        click_count = 0
        
        # 국어 탭 셀렉터
        korean_tab_selectors = [
            "text='국어'",
            "a:has-text('국어')",
            "//a[contains(text(), '국어')]",
            "li:has-text('국어') a",
        ]
        
        try:
            while True:
                elapsed = time.time() - start_time
                
                # 타임아웃 체크
                if elapsed > max_duration:
                    self.log(f"⏱️ 타임아웃: {max_duration}초 경과 ({click_count}회 클릭)", "WARNING")
                    return None
                
                # 국어 탭 클릭 시도
                korean_tab_clicked = False
                for selector in korean_tab_selectors:
                    try:
                        element = self.page.wait_for_selector(selector, timeout=100)
                        if element and element.is_visible():
                            element.click(timeout=100)
                            click_count += 1
                            korean_tab_clicked = True
                            break
                    except:
                        continue
                
                if not korean_tab_clicked:
                    self.log("국어 탭을 찾을 수 없습니다.", "WARNING")
                
                # 100회마다 로그
                if click_count % 100 == 0:
                    self.log(f"국어 탭 클릭 횟수: {click_count}회 (경과: {elapsed:.1f}초)")
                
                # 결제하기 버튼이 나타났는지 체크
                try:
                    all_buttons = self.page.query_selector_all('text="결제하기"')
                    
                    if len(all_buttons) > 0:
                        self.log(f"✅ '결제하기' 버튼 발견! (국어 탭 클릭 {click_count}회, {elapsed:.3f}초)")
                        
                        # 최적의 버튼 찾기
                        target_button = None
                        best_match_score = 0
                        
                        for idx, button in enumerate(all_buttons):
                            parent = button.evaluate_handle('el => el.closest("tr, div, li")')
                            if parent:
                                parent_text = parent.evaluate('el => el.innerText')
                                
                                match_score = 0
                                if teacher_name in parent_text:
                                    match_score += 1
                                    if course_name in parent_text:
                                        match_score += 10
                                    else:
                                        if "[정규/LIVE]" in course_name and "[정규/LIVE]" in parent_text:
                                            match_score += 5
                                        elif "[정규]" in course_name and "[정규]" in parent_text and "[정규/LIVE]" not in parent_text:
                                            match_score += 5
                                        for time_keyword in ["(오전반)", "(오후반)", "(종일반)", "(고3반)"]:
                                            if time_keyword in course_name and time_keyword in parent_text:
                                                match_score += 2
                                                break
                                
                                if match_score > best_match_score:
                                    best_match_score = match_score
                                    target_button = button
                        
                        if target_button and best_match_score > 0:
                            self.log(f"✅ 목표 '결제하기' 버튼 찾음! (매칭 점수: {best_match_score})")
                            return target_button
                        else:
                            self.log("조건에 맞는 '결제하기' 버튼을 찾지 못했습니다. 계속 시도...")
                
                except:
                    pass
                
                # 클릭 간격 대기
                time.sleep(click_interval)
        
        except KeyboardInterrupt:
            self.log("\n사용자에 의해 중단되었습니다.", "WARNING")
            return None
        
        except Exception as e:
            self.log(f"오류 발생: {e}", "ERROR")
            return None
    
    def rapid_click_until_change(self, button_selector: str = None, button_element = None,
                                 max_duration: int = 30, click_interval: float = 0.001):
        """버튼을 초고속으로 연속 클릭하면서 페이지 변화 감지
        
        Args:
            button_selector: 클릭할 버튼의 CSS 셀렉터 (또는 button_element 사용)
            button_element: 클릭할 버튼 요소 (selector보다 우선)
            max_duration: 최대 클릭 시도 시간 (초)
            click_interval: 클릭 간격 (초, 기본 0.001 = 1ms)
        
        Returns:
            bool: 페이지 변화 감지 시 True, 타임아웃 시 False
        """
        self.log("="*80)
        self.log("🚀 초고속 연속 클릭 시작!")
        self.log(f"설정: 클릭 간격 {click_interval*1000:.1f}ms, 최대 {max_duration}초")
        self.log("="*80)
        
        # 초기 URL 및 페이지 상태 기록
        initial_url = self.page.url
        initial_page_count = len(self.context.pages)
        
        # 변화 감지 플래그
        change_detected = False
        change_type = None
        
        start_time = time.time()
        click_count = 0
        
        try:
            # 버튼 요소 준비
            if not button_element:
                if button_selector:
                    button_element = self.page.wait_for_selector(button_selector, timeout=5000)
                else:
                    self.log("버튼 셀렉터 또는 요소가 필요합니다.", "ERROR")
                    return False
            
            self.log(f"버튼 확인 완료. 연속 클릭을 시작합니다...")
            
            while True:
                elapsed = time.time() - start_time
                
                # 타임아웃 체크
                if elapsed > max_duration:
                    self.log(f"⏱️ 타임아웃: {max_duration}초 경과 ({click_count}회 클릭)", "WARNING")
                    break
                
                try:
                    # 버튼 클릭
                    button_element.click(timeout=100)
                    click_count += 1
                    
                    # 10회마다 로그 (너무 많은 로그 방지)
                    if click_count % 100 == 0:
                        self.log(f"클릭 횟수: {click_count}회 (경과: {elapsed:.1f}초)")
                    
                    # 페이지 변화 체크 (매 클릭마다)
                    # 1. 새 페이지/팝업 생성 체크
                    current_page_count = len(self.context.pages)
                    if current_page_count > initial_page_count:
                        change_detected = True
                        change_type = "새 팝업 윈도우 감지"
                        self.log(f"✅ {change_type}! (클릭 {click_count}회, {elapsed:.3f}초)")
                        break
                    
                    # 2. URL 변경 체크
                    current_url = self.page.url
                    if current_url != initial_url:
                        change_detected = True
                        change_type = "URL 변경 감지"
                        self.log(f"✅ {change_type}!")
                        self.log(f"   이전: {initial_url}")
                        self.log(f"   현재: {current_url}")
                        self.log(f"   클릭 횟수: {click_count}회, 소요 시간: {elapsed:.3f}초")
                        break
                    
                    # 3. 특정 성공 메시지 또는 에러 메시지 체크 (옵션)
                    # 페이지 내용에 "접수 완료" 또는 "성공" 같은 텍스트가 있는지 체크
                    try:
                        # 빠른 체크를 위해 timeout을 매우 짧게 설정
                        success_indicators = [
                            'text="접수 완료"',
                            'text="접수가 완료"',
                            'text="신청 완료"',
                            'text="성공"',
                            'text="접수 가능"',
                            'text="결제"',
                            'text="결제 가능"',
                        ]
                        
                        for indicator in success_indicators:
                            if self.page.query_selector(indicator):
                                change_detected = True
                                change_type = f"성공 메시지 감지: {indicator}"
                                self.log(f"✅ {change_type}! (클릭 {click_count}회, {elapsed:.3f}초)")
                                break
                        
                        if change_detected:
                            break
                    except:
                        pass  # 성공 메시지 체크는 선택사항
                    
                    # 클릭 간격 대기
                    time.sleep(click_interval)
                    
                except Exception as click_error:
                    # 클릭 에러는 무시하고 계속 (버튼이 일시적으로 사라질 수 있음)
                    if click_count % 1000 == 0:  # 가끔씩만 로그
                        self.log(f"클릭 에러 (무시): {click_error}", "DEBUG")
                    time.sleep(click_interval)
                    continue
        
        except KeyboardInterrupt:
            self.log("\n사용자에 의해 중단되었습니다.", "WARNING")
            return False
        
        # 결과 요약
        self.log("="*80)
        if change_detected:
            self.log(f"✅ 페이지 변화 감지 성공!")
            self.log(f"   변화 유형: {change_type}")
            self.log(f"   총 클릭 횟수: {click_count}회")
            self.log(f"   소요 시간: {elapsed:.3f}초")
            self.log(f"   평균 클릭 속도: {click_count/elapsed:.1f}회/초")
        else:
            self.log(f"❌ 타임아웃")
            self.log(f"   총 클릭 횟수: {click_count}회")
            self.log(f"   소요 시간: {elapsed:.1f}초")
        self.log("="*80)
        
        return change_detected
    
    def run_rapid_registration(self, username: str = None, password: str = None,
                              teacher_name: str = "강민철",
                              course_name: str = "[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)",
                              target_time: str = "19:00",
                              click_interval: float = 0.001,
                              max_duration: int = 30,
                              start_immediately: bool = False,
                              screenshot: bool = True):
        """실시간 수강신청 모드 실행
        
        특정 시간(기본 19:00)까지 대기한 후, 초고속으로 버튼을 연속 클릭하여
        접수 상태로 변경되는 순간 즉시 접수
        
        Args:
            username: 로그인 아이디
            password: 로그인 비밀번호
            teacher_name: 강사 이름
            course_name: 강의명
            target_time: 대기할 목표 시간 (예: "19:00", "19:00:00", None이면 즉시 시작)
            click_interval: 클릭 간격 (초)
            max_duration: 최대 클릭 시도 시간 (초)
            start_immediately: True면 시간 대기 없이 즉시 클릭 시작
            screenshot: 스크린샷 저장 여부
        """
        self.log("="*80)
        self.log("🎯 실시간 수강신청 모드 시작")
        self.log("="*80)
        
        # 1단계: 로그인
        if username and password:
            self.log("\n[1단계] 로그인 중...")
            if not self.login(username, password):
                self.log("로그인 실패", "ERROR")
                return False
            if screenshot:
                self.save_screenshot("step0_login_success.png")
        
        # 2단계: 사이트 방문
        self.log("\n[2단계] 사이트 방문 중...")
        if not self.visit_site():
            self.log("사이트 방문 실패", "ERROR")
            return False
        if screenshot:
            self.save_screenshot("step1_initial.png")
        
        # 3단계: 목표 시간까지 대기 (옵션)
        if start_immediately:
            self.log(f"\n[3단계] ⚡ 즉시 시작 모드 - 시간 대기 건너뛰기")
            self.log(f"현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            self.log(f"\n[3단계] 목표 시간({target_time})까지 대기 중...")
            self.wait_until_time(target_time)
        
        # 4단계: 국어 탭을 계속 클릭하면서 결제하기 버튼 출현 대기
        self.log("\n[4단계] 국어 탭 클릭 → 결제하기 버튼 출현 대기")
        target_button = self.rapid_click_korean_tab_until_payment_button(
            teacher_name=teacher_name,
            course_name=course_name,
            max_duration=max_duration,
            click_interval=click_interval
        )
        
        if not target_button:
            self.log("결제하기 버튼을 찾지 못했습니다.", "ERROR")
            if screenshot:
                self.save_screenshot("step4_button_not_found.png")
            return False
        
        if screenshot:
            self.save_screenshot("step4_button_found.png")
        
        # 5단계: 결제하기 버튼 초고속 연속 클릭
        self.log("\n[5단계] 결제하기 버튼 초고속 연속 클릭 시작!")
        success = self.rapid_click_until_change(
            button_element=target_button,
            max_duration=max_duration,
            click_interval=click_interval
        )
        
        if success:
            self.log("\n✅ 결제 페이지 접근 성공! 페이지 변화를 감지했습니다.")
            if screenshot:
                self.save_screenshot("step5_success.png")
            
            # 팝업이 있으면 캡처
            try:
                if len(self.context.pages) > 1:
                    popup = self.context.pages[-1]
                    screenshot_dir = Path('screenshots')
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    popup_screenshot = screenshot_dir / f"step6_popup_{timestamp}.png"
                    popup.screenshot(path=str(popup_screenshot), full_page=True)
                    self.log(f"팝업 스크린샷 저장: {popup_screenshot}")
            except:
                pass
            
            self.log("="*80)
            self.log("🎉 실시간 수강신청 완료!")
            self.log("="*80)
            return True
        else:
            self.log("\n⚠️ 타임아웃: 페이지 변화를 감지하지 못했습니다.", "WARNING")
            if screenshot:
                self.save_screenshot("step5_timeout.png")
            return False
    
    def run(self, username: str = None, password: str = None,
            teacher_name: str = "강민철", 
            course_name: str = "[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)",
            screenshot: bool = True):
        """전체 매크로 실행"""
        self.log("="*80)
        self.log("메가스터디 러셀 단과 접수 자동화 시작")
        self.log("="*80)
        
        # 0단계: 로그인 (옵션)
        if username and password:
            self.log("로그인 단계 시작")
            if not self.login(username, password):
                self.log("매크로 실행 실패: 로그인 단계", "ERROR")
                if screenshot:
                    self.save_screenshot("step0_login_error.png")
                return False
            
            if screenshot:
                self.save_screenshot("step0_login_success.png")
        else:
            self.log("로그인 정보가 제공되지 않았습니다. 로그인 단계를 건너뜁니다.", "WARNING")
        
        # 1단계: 사이트 방문
        if not self.visit_site():
            self.log("매크로 실행 실패: 사이트 방문 단계", "ERROR")
            return False
        
        if screenshot:
            self.save_screenshot("step1_initial.png")
        
        # 2단계: 국어 탭 클릭
        if not self.click_korean_tab():
            self.log("매크로 실행 실패: 국어 탭 클릭 단계", "ERROR")
            if screenshot:
                self.save_screenshot("step2_error.png")
            return False
        
        if screenshot:
            self.save_screenshot("step2_korean_tab.png")
        
        # 3단계: 결제하기 버튼 클릭
        if not self.click_registration_button(teacher_name, course_name):
            self.log("매크로 실행 실패: 결제하기 버튼 클릭 단계", "ERROR")
            if screenshot:
                self.save_screenshot("step3_error.png")
            return False
        
        if screenshot:
            self.save_screenshot("step3_registration.png")
        
        self.log("="*80)
        self.log("매크로 실행 완료!")
        self.log("="*80)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='메가스터디 러셀 단과 접수 자동화 매크로',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
사용 예시:
  # 로그인 포함 기본 실행
  python russel_macro.py --username "아이디" --password "비밀번호"
  
  # 환경 변수로 로그인 정보 제공
  export RUSSEL_USERNAME="아이디"
  export RUSSEL_PASSWORD="비밀번호"
  python russel_macro.py
  
  # 대화형 로그인 (비밀번호 숨김)
  python russel_macro.py --interactive
  
  # 입력과 클릭 동작을 천천히 보기 (디버깅/데모용)
  python russel_macro.py --slow-mo 1000 --typing-delay 200
  
  # 헤드리스 모드 (백그라운드 실행)
  python russel_macro.py --username "아이디" --password "비밀번호" --headless
  
  # 빠른 실행
  python russel_macro.py --username "아이디" --password "비밀번호" --fast
  
  # 특정 강사/강의 지정
  python russel_macro.py --username "아이디" --password "비밀번호" --teacher "강민철" --course "[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)"
  
  # 로그인 없이 실행 (로그인이 필요 없는 경우)
  python russel_macro.py --no-login
        '''
    )
    
    parser.add_argument('--username', '-u',
                       help='로그인 아이디 (환경 변수: RUSSEL_USERNAME)')
    parser.add_argument('--password', '-p',
                       help='로그인 비밀번호 (환경 변수: RUSSEL_PASSWORD)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='대화형 로그인 (비밀번호 숨김 입력)')
    parser.add_argument('--no-login', action='store_true',
                       help='로그인 단계 건너뛰기')
    parser.add_argument('--headless', action='store_true',
                       help='헤드리스 모드 (브라우저 창 숨김)')
    parser.add_argument('--fast', action='store_true',
                       help='빠른 실행 (slow_mo=0)')
    parser.add_argument('--teacher', default='강민철',
                       help='강사 이름 (기본값: 강민철)')
    parser.add_argument('--course', default='[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)',
                       help='강의명 (기본값: [정규][독서·문학] 2027 강민철의 기출 분석 (고3반))')
    parser.add_argument('--no-screenshot', action='store_true',
                       help='스크린샷 저장 안 함')
    parser.add_argument('--slow-mo', type=int, default=1,
                       help='작업 속도 조절 (밀리초, 기본값: 1)')
    parser.add_argument('--typing-delay', type=int, default=1,
                       help='타이핑 속도 조절 (밀리초/글자, 기본값: 1)')
    parser.add_argument('--keep-open', action='store_true',
                       help='완료 후 브라우저를 열어둠')
    parser.add_argument('--rapid-mode', action='store_true',
                       help='🚀 실시간 수강신청 모드 (특정 시간에 초고속 연속 클릭)')
    parser.add_argument('--target-time', default='18:59',
                       help='실시간 수강신청 목표 시간 (기본값: 19:00, 형식: HH:MM 또는 HH:MM:SS)')
    parser.add_argument('--click-interval', type=float, default=0.001,
                       help='연속 클릭 간격 (초, 기본값: 0.001 = 1ms)')
    parser.add_argument('--max-click-duration', type=int, default=300,
                       help='최대 클릭 시도 시간 (초, 기본값: 300)')
    parser.add_argument('--start-immediately', action='store_true',
                       help='⚡ 시간 대기 없이 즉시 클릭 시작 (rapid-mode에서만 유효)')
    
    args = parser.parse_args()
    
    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright가 설치되어 있지 않습니다.")
        print("설치 방법: pip install playwright && playwright install")
        return 1
    
    # 로그인 정보 처리
    username = None
    password = None
    
    if not args.no_login:
        if args.interactive:
            # 대화형 입력
            print("로그인 정보를 입력하세요:")
            username = input("아이디: ").strip()
            password = getpass.getpass("비밀번호: ")
        else:
            # 커맨드라인 인자 우선, 없으면 환경 변수
            username = args.username or os.getenv('RUSSEL_USERNAME')
            password = args.password or os.getenv('RUSSEL_PASSWORD')
            
            if not username or not password:
                print("\n⚠️  로그인 정보가 제공되지 않았습니다.")
                print("\n다음 방법 중 하나를 선택하세요:")
                print("  1. 커맨드라인 인자: --username <아이디> --password <비밀번호>")
                print("  2. 환경 변수: export RUSSEL_USERNAME=<아이디> RUSSEL_PASSWORD=<비밀번호>")
                print("  3. 대화형 입력: --interactive")
                print("  4. 로그인 없이 실행: --no-login")
                
                # 대화형으로 전환할지 물어보기
                choice = input("\n대화형으로 로그인 정보를 입력하시겠습니까? (y/n): ").strip().lower()
                if choice == 'y':
                    username = input("아이디: ").strip()
                    password = getpass.getpass("비밀번호: ")
                else:
                    print("\n로그인 없이 계속 진행합니다.")
                    username = None
                    password = None
    
    # slow_mo 설정
    slow_mo = 0 if args.fast else args.slow_mo
    typing_delay = args.typing_delay
    
    # rapid-mode에서는 자동으로 keep_open 활성화
    # 일반 모드에서도 기본적으로 브라우저를 열어둠 (결제 페이지 확인을 위해)
    keep_open = True if not args.headless else args.keep_open
    
    # 매크로 실행
    try:
        with RusselMacro(headless=args.headless, slow_mo=slow_mo, typing_delay=typing_delay, keep_open=keep_open) as macro:
            
            # 실시간 수강신청 모드
            if args.rapid_mode:
                print("\n" + "="*80)
                print("🚀 실시간 수강신청 모드")
                print("="*80)
                if args.start_immediately:
                    print("⚡ 시간 대기 없이 즉시 시작")
                else:
                    print(f"목표 시간: {args.target_time}")
                print(f"클릭 간격: {args.click_interval*1000:.1f}ms")
                print(f"최대 클릭 시간: {args.max_click_duration}초")
                print("="*80 + "\n")
                
                success = macro.run_rapid_registration(
                    username=username,
                    password=password,
                    teacher_name=args.teacher,
                    course_name=args.course,
                    target_time=args.target_time,
                    click_interval=args.click_interval,
                    max_duration=args.max_click_duration,
                    start_immediately=args.start_immediately,
                    screenshot=not args.no_screenshot
                )
            
            # 일반 모드
            else:
                success = macro.run(
                    username=username,
                    password=password,
                    teacher_name=args.teacher,
                    course_name=args.course,
                    screenshot=not args.no_screenshot
                )
            
            if success:
                print("\n✅ 매크로 실행 성공!")
                return 0
            else:
                print("\n❌ 매크로 실행 실패!")
                return 1
                
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        return 130
    except Exception as e:
        print(f"\n❌ 예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

