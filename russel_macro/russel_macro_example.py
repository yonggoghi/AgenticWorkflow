#!/usr/bin/env python3
"""
메가스터디 러셀 매크로 실행 예제

이 스크립트는 russel_macro.py의 사용 예제를 보여줍니다.
"""

import os
import getpass
from russel_macro import RusselMacro

def get_credentials():
    """로그인 정보 입력"""
    print("\n로그인 정보를 입력하세요 (환경 변수 RUSSEL_USERNAME, RUSSEL_PASSWORD 사용 가능):")
    username = os.getenv('RUSSEL_USERNAME') or input("아이디: ").strip()
    password = os.getenv('RUSSEL_PASSWORD') or getpass.getpass("비밀번호: ")
    return username, password

def example_basic():
    """기본 실행 예제"""
    print("\n" + "="*80)
    print("예제 1: 기본 실행 (로그인 포함)")
    print("="*80 + "\n")
    
    username, password = get_credentials()
    
    # keep_open=True로 설정하면 완료 후 브라우저가 열려있음
    with RusselMacro(headless=False, slow_mo=500, keep_open=True) as macro:
        success = macro.run(
            username=username,
            password=password,
            teacher_name="강민철",
            course_name="[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)",
            screenshot=True
        )
        
        if success:
            print("\n✅ 매크로 실행 성공!")
        else:
            print("\n❌ 매크로 실행 실패!")


def example_headless():
    """헤드리스 모드 실행 예제"""
    print("\n" + "="*80)
    print("예제 2: 헤드리스 모드 (백그라운드 실행)")
    print("="*80 + "\n")
    
    username, password = get_credentials()
    
    with RusselMacro(headless=True, slow_mo=0) as macro:
        success = macro.run(
            username=username,
            password=password,
            teacher_name="강민철",
            course_name="[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)",
            screenshot=True
        )
        
        if success:
            print("\n✅ 매크로 실행 성공!")
        else:
            print("\n❌ 매크로 실행 실패!")


def example_step_by_step():
    """단계별 실행 예제"""
    print("\n" + "="*80)
    print("예제 3: 단계별 실행 (커스텀 제어)")
    print("="*80 + "\n")
    
    username, password = get_credentials()
    
    with RusselMacro(headless=False, slow_mo=1000) as macro:
        # 0단계: 로그인
        if username and password:
            if not macro.login(username, password):
                print("❌ 로그인 실패")
                return
            macro.save_screenshot("custom_step0_login.png")
            print("✅ 0단계 완료: 로그인")
        
        # 1단계: 사이트 방문
        if not macro.visit_site():
            print("❌ 사이트 방문 실패")
            return
        
        macro.save_screenshot("custom_step1.png")
        print("✅ 1단계 완료: 사이트 방문")
        
        # 2단계: 국어 탭 클릭
        if not macro.click_korean_tab():
            print("❌ 국어 탭 클릭 실패")
            return
        
        macro.save_screenshot("custom_step2.png")
        print("✅ 2단계 완료: 국어 탭 클릭")
        
        # 3단계: 결제하기 버튼 클릭
        if not macro.click_registration_button("강민철", "[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)"):
            print("❌ 결제하기 버튼 클릭 실패")
            return
        
        macro.save_screenshot("custom_step3.png")
        print("✅ 3단계 완료: 결제하기 버튼 클릭")
        
        print("\n✅ 전체 프로세스 완료!")


def example_multiple_courses():
    """여러 강의 순회 예제"""
    print("\n" + "="*80)
    print("예제 4: 여러 강의 접수 대기 (순차 실행)")
    print("="*80 + "\n")
    
    username, password = get_credentials()
    
    courses = [
        {"teacher": "강민철", "course": "[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)"},
        # 추가 강의가 있다면 여기에 추가
        # {"teacher": "다른 강사", "course": "다른 강의"},
    ]
    
    for idx, course_info in enumerate(courses, 1):
        print(f"\n강의 {idx}/{len(courses)}: {course_info['teacher']} - {course_info['course']}")
        
        with RusselMacro(headless=True, slow_mo=0) as macro:
            success = macro.run(
                username=username,
                password=password,
                teacher_name=course_info['teacher'],
                course_name=course_info['course'],
                screenshot=True
            )
            
            if success:
                print(f"✅ 강의 {idx} 접수 완료")
            else:
                print(f"❌ 강의 {idx} 접수 실패")


def example_rapid_registration():
    """실시간 수강신청 모드 예제"""
    print("\n" + "="*80)
    print("예제 5: 🚀 실시간 수강신청 모드 (권장: 미리 시작)")
    print("="*80 + "\n")
    
    username, password = get_credentials()
    
    print("⚠️  핵심 전략:")
    print("- 로컬 시간과 서버 시간이 다를 수 있으므로 2-3분 일찍 시작하세요!")
    print("- 19:00 결제 가능 → 18:58부터 클릭 권장")
    print("- 시간이 되면 1ms 간격으로 초고속 연속 클릭을 시작합니다.")
    print("- 페이지 변화 감지 시 자동으로 중단하고 브라우저를 유지합니다.")
    print()
    
    print("옵션을 선택하세요:")
    print("1. 특정 시간부터 시작 (권장: 결제 가능 시간보다 2-3분 일찍)")
    print("2. 즉시 시작 (시간 대기 없음)")
    choice = input("\n선택 (1-2): ").strip()
    
    start_immediately = False
    target_time = "18:58"
    
    if choice == "2":
        start_immediately = True
    else:
        target_time = input("목표 시간을 입력하세요 (기본값: 18:58, 19:00 결제 가능 대비): ").strip() or "18:58"
    
    # 실시간 수강신청 모드는 자동으로 keep_open=True
    with RusselMacro(headless=False, slow_mo=0, keep_open=True) as macro:
        success = macro.run_rapid_registration(
            username=username,
            password=password,
            teacher_name="강민철",
            course_name="[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)",
            target_time=target_time,
            click_interval=0.001,  # 1ms
            max_duration=180,  # 최대 180초 (3분)
            start_immediately=start_immediately,
            screenshot=True
        )
        
        if success:
            print("\n✅ 실시간 수강신청 성공!")
        else:
            print("\n❌ 실시간 수강신청 실패!")


def example_rapid_registration_test():
    """실시간 수강신청 테스트 모드 (즉시 클릭)"""
    print("\n" + "="*80)
    print("예제 6: 🧪 실시간 수강신청 테스트 (즉시 시작)")
    print("="*80 + "\n")
    
    username, password = get_credentials()
    
    print("⚠️  테스트 모드:")
    print("- 시간 대기 없이 즉시 버튼을 찾아서 초고속 클릭을 시작합니다.")
    print("- 실제 사용 전에 동작을 테스트하는 용도입니다.")
    print("- 5초간만 클릭을 시도합니다.")
    print()
    
    with RusselMacro(headless=False, slow_mo=0, keep_open=True) as macro:
        success = macro.run_rapid_registration(
            username=username,
            password=password,
            teacher_name="강민철",
            course_name="[정규][독서·문학] 2027 강민철의 기출 분석 (고3반)",
            target_time="19:00",  # 사용되지 않음 (start_immediately=True)
            click_interval=0.001,
            max_duration=5,  # 테스트는 5초만
            start_immediately=True,  # 즉시 시작
            screenshot=True
        )
        
        if success:
            print("\n✅ 테스트 성공!")
        else:
            print("\n⚠️  테스트 완료 (변화 감지 안 됨)")


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("메가스터디 러셀 매크로 실행 예제")
    print("="*80)
    
    print("\n실행할 예제를 선택하세요:")
    print("1. 기본 실행 (브라우저 표시)")
    print("2. 헤드리스 모드 (백그라운드 실행)")
    print("3. 단계별 실행 (커스텀 제어)")
    print("4. 여러 강의 순회")
    print("5. 🚀 실시간 수강신청 모드 (19:00 자동 클릭)")
    print("6. 🧪 실시간 수강신청 테스트 (즉시 클릭)")
    print("0. 종료")
    
    choice = input("\n선택 (0-6): ").strip()
    
    if choice == "1":
        example_basic()
    elif choice == "2":
        example_headless()
    elif choice == "3":
        example_step_by_step()
    elif choice == "4":
        example_multiple_courses()
    elif choice == "5":
        example_rapid_registration()
    elif choice == "6":
        example_rapid_registration_test()
    elif choice == "0":
        print("프로그램을 종료합니다.")
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

