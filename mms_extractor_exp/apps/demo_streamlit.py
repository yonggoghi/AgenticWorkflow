import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional
import base64
from pathlib import Path
import pandas as pd
import argparse
import sys
import os
# Add parent directory to path to allow imports from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MongoDB 유틸리티는 필요할 때 동적으로 임포트

# 페이지 설정
st.set_page_config(
    page_title="MMS Extractor API Demo",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sample-message {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4f46e5;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .sample-message:hover {
        background: #e2e8f0;
    }
    .status-online {
        color: #10b981;
        font-weight: bold;
    }
    .status-offline {
        color: #ef4444;
        font-weight: bold;
    }
    .result-container {
        background: #f1f5f9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 커맨드라인 인자 파싱 (Streamlit 호환)
def parse_args():
    import sys
    
    # 기본값 설정
    api_port = 8000
    demo_port = 8082
    
    # sys.argv에서 우리가 원하는 인수만 찾기
    args_list = sys.argv[1:]  # 스크립트 이름 제외
    
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        
        if arg == '--api-port':
            if i + 1 < len(args_list):
                try:
                    api_port = int(args_list[i + 1])
                    i += 1  # 값도 건너뛰기
                except ValueError:
                    pass
        elif arg == '--demo-port':
            if i + 1 < len(args_list):
                try:
                    demo_port = int(args_list[i + 1])
                    i += 1  # 값도 건너뛰기
                except ValueError:
                    pass
        elif arg.startswith('--api-port='):
            try:
                api_port = int(arg.split('=', 1)[1])
            except ValueError:
                pass
        elif arg.startswith('--demo-port='):
            try:
                demo_port = int(arg.split('=', 1)[1])
            except ValueError:
                pass
        
        i += 1
    
    # 간단한 객체로 반환 (argparse.Namespace와 유사)
    class Args:
        def __init__(self, api_port, demo_port):
            self.api_port = api_port
            self.demo_port = demo_port
    
    return Args(api_port, demo_port)

# 인자 파싱
args = parse_args()

# API 설정
API_BASE_URL = f"http://localhost:{args.api_port}"  # MMS Extractor API
DEMO_API_BASE_URL = f"http://localhost:{args.demo_port}"  # Demo Server API

# 샘플 메시지 데이터
SAMPLE_MESSAGES = [
    {
        "title": "[SK텔레콤] ZEM폰 포켓몬에디션3 안내 - 우리 아이 첫 번째 스마트폰",
        "content": """[SK텔레콤] ZEM폰 포켓몬에디션3 안내
(광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!
우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,
ZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!

✨ 특별 혜택
- 월 요금 20% 할인 (첫 6개월)
- 포켓몬 케이스 무료 증정
- ZEM 프리미엄 서비스 3개월 무료

📞 문의: 1588-0011 (평일 9시-18시)
🏪 가까운 T world 매장 방문
🌐 www.tworld.co.kr

수신거부 080-011-0000"""
    },
    {
        "title": "[T world] 5G 요금제 특가 혜택 - 월 39,000원 할인",
        "content": """[T world] 5G 요금제 특가 혜택
(광고) 5G 슈퍼플랜 특가 이벤트 진행 중!

🎯 이달의 특가
- 5G 슈퍼플랜 (데이터 무제한): 월 79,000원 → 39,000원 (50% 할인)
- 가족 추가 회선: 월 29,000원
- YouTube Premium 6개월 무료

📅 이벤트 기간: 2024.01.01 ~ 2024.01.31
🎁 신규 가입 시 갤럭시 버즈 증정

온라인 가입: m.tworld.co.kr
매장 방문: 전국 T world 매장
고객센터: 114

수신거부 080-011-0000"""
    },
    {
        "title": "[SK텔레콤] 갤럭시 S24 사전예약 - 최대 30만원 할인",
        "content": """[SK텔레콤] 갤럭시 S24 사전예약
(광고) 갤럭시 S24 시리즈 사전예약 시작!

🌟 사전예약 혜택
- 갤럭시 S24 Ultra: 최대 30만원 할인
- 갤럭시 S24+: 최대 25만원 할인  
- 갤럭시 S24: 최대 20만원 할인

🎁 추가 혜택
- 갤럭시 워치6 50% 할인
- 무선충전기 무료 증정
- 케어플러스 6개월 무료

📅 사전예약: 2024.01.10 ~ 2024.01.24
📱 출시일: 2024.01.31

T world 앱에서 간편 예약
매장 예약: tworldfriends.co.kr/D123456789
문의: 1588-0011

수신거부 080-011-0000"""
    },
    {
        "title": "[T 우주] 넷플릭스와 웨이브를 월 9,900원에!",
        "content": """(광고)[SKT] 넷플릭스+웨이브 월 9,900원, 이게 되네! 
#04 고객님,
넷플릭스와 웨이브 둘 다 보고 싶었지만, 가격 때문에 망설이셨다면 지금이 바로 기회! 
오직 T 우주에서만, 
2개월 동안 월 9,900원에 넷플릭스와 웨이브를 모두 즐기실 수 있습니다.
8월 31일까지만 드리는 혜택이니, 지금 바로 가입해 보세요! 

■ 우주패스 Netflix 런칭 프로모션 
- 기간 : 2024년 8월 31일(토)까지
- 혜택 : 우주패스 Netflix(광고형 스탠다드)를 2개월 동안 월 9,900원에 이용 가능한 쿠폰 제공

▶ 프로모션 자세히 보기: http://t-mms.kr/jAs/#74

■ 우주패스 Netflix(월 12,000원)  
- 기본 혜택 : Netflix 광고형 스탠다드 멤버십
- 추가 혜택 : Wavve 콘텐츠 팩 
* 추가 요금을 내시면 Netflix 스탠다드와 프리미엄 멤버십 상품으로 가입 가능합니다.  

■ 유의 사항
-  프로모션 쿠폰은 1인당 1회 다운로드 가능합니다. 
-  쿠폰 할인 기간이 끝나면 정상 이용금액으로 자동 결제 됩니다. 

■ 문의: T 우주 고객센터 (1505, 무료)
나만의 구독 유니버스, T 우주 
무료 수신거부 1504
"""
    }
]

def check_api_status() -> bool:
    """API 서버 상태 확인"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_extraction_api(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """MMS 추출 API 호출"""
    try:
        st.write(f"🔍 API 호출 중: {API_BASE_URL}/extract")
        st.write(f"📤 전송 데이터: {data}")
        
        # 타임아웃을 120초로 증가 (LLM 처리 시간 고려)
        response = requests.post(
            f"{API_BASE_URL}/extract",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2분으로 증가
        )
        
        if response.status_code == 200:
            result = response.json()
            st.write("✅ API 응답 성공!")
            
            # 디버깅: 응답 구조 확인
            st.write("🔍 응답 구조 확인:")
            st.write(f"응답 키들: {list(result.keys())}")
            
            # 전체 응답 데이터는 너무 클 수 있으므로 요약만 표시
            if 'success' in result:
                st.write(f"처리 성공: {result.get('success')}")
            if 'metadata' in result:
                metadata = result['metadata']
                st.write(f"처리 시간: {metadata.get('processing_time_seconds', 'N/A')}초")
                
            # extracted_data가 있는지 확인
            if 'extracted_data' in result:
                extracted_data = result['extracted_data']
                st.write(f"추출된 데이터 키들: {list(extracted_data.keys())}")
            else:
                st.write("⚠️ extracted_data가 응답에 없습니다.")
                
            return result
        else:
            st.error(f"❌ API 응답 오류: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ReadTimeout:
        st.error("⏰ API 호출 시간 초과 (2분)")
        st.error("API 서버가 응답하는데 시간이 오래 걸리고 있습니다. 다시 시도해보세요.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 API 서버에 연결할 수 없습니다.")
        st.error("API 서버가 실행 중인지 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"API 호출 오류: {str(e)}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        return None

def call_prompts_api(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """프롬프트 구성 API 호출"""
    try:
        # 디버깅 정보는 expander 안에 숨김
        with st.expander("🔍 프롬프트 API 호출 디버깅"):
            st.write(f"🔍 프롬프트 API 호출 중: {API_BASE_URL}/prompts")
            st.write(f"📤 전송 데이터: {data}")
        
        response = requests.post(
            f"{API_BASE_URL}/prompts",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=60  # 타임아웃 증가 (실제 추출을 수행하므로)
        )
        
        with st.expander("🔍 프롬프트 API 응답 디버깅"):
            st.write(f"📥 응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            with st.expander("🔍 프롬프트 API 응답 디버깅"):
                st.write(f"✅ 프롬프트 API 응답 성공!")
                st.write(f"🔍 응답 키들: {list(result.keys())}")
                st.write(f"📊 성공 여부: {result.get('success', 'N/A')}")
                if 'prompts' in result:
                    st.write(f"📝 프롬프트 개수: {len(result['prompts'])}")
                    st.write(f"📝 프롬프트 키들: {list(result['prompts'].keys())}")
            return result
        else:
            st.error(f"프롬프트 API 호출 실패: {response.status_code}")
            st.error(f"응답 내용: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"프롬프트 API 호출 오류: {str(e)}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        return None

def display_prompts(prompts_data: Dict[str, Any]):
    """프롬프트 표시"""
    if not prompts_data or not prompts_data.get('success'):
        st.error("프롬프트 데이터를 가져올 수 없습니다.")
        if prompts_data and 'error' in prompts_data:
            st.error(f"오류: {prompts_data['error']}")
        return
    
    prompts = prompts_data.get('prompts', {})
    settings = prompts_data.get('settings', {})
    
    # 설정 정보 표시
    with st.expander("⚙️ 현재 설정 정보"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**LLM 모델**: {settings.get('llm_model', 'N/A')}")
            st.write(f"**데이터 소스**: {settings.get('offer_info_data_src', 'N/A')}")
            st.write(f"**상품 추출 모드**: {settings.get('product_info_extraction_mode', 'N/A')}")
        with col2:
            st.write(f"**엔티티 매칭 모드**: {settings.get('entity_matching_mode', 'N/A')}")
            st.write(f"**DAG 추출**: {'활성화' if settings.get('extract_entity_dag', False) else '비활성화'}")
    
    # 프롬프트 표시 순서 정의
    prompt_order = [
        'main_extraction_prompt',  # 메인 정보 추출
        'entity_extraction_prompt',  # 엔티티 추출
        'dag_extraction_prompt'  # DAG 관계 추출
    ]
    
    # 정의된 순서대로 프롬프트 표시
    for prompt_key in prompt_order:
        if prompt_key in prompts:
            prompt_info = prompts[prompt_key]
            with st.expander(f"📝 {prompt_info.get('title', prompt_key)}"):
                st.write(f"**설명**: {prompt_info.get('description', '설명 없음')}")
                st.write(f"**길이**: {prompt_info.get('length', 0):,} 문자")
                
                # 프롬프트 내용을 코드 블록으로 표시
                prompt_content = prompt_info.get('content', '')
                if prompt_content and not prompt_content.startswith('오류:'):
                    st.code(prompt_content, language='text')
                else:
                    st.error("프롬프트 내용을 표시할 수 없습니다.")
    
    # 순서에 정의되지 않은 추가 프롬프트가 있으면 마지막에 표시
    for prompt_key, prompt_info in prompts.items():
        if prompt_key not in prompt_order:
            with st.expander(f"📝 {prompt_info.get('title', prompt_key)}"):
                st.write(f"**설명**: {prompt_info.get('description', '설명 없음')}")
                st.write(f"**길이**: {prompt_info.get('length', 0):,} 문자")
                
                # 프롬프트 내용을 코드 블록으로 표시
                prompt_content = prompt_info.get('content', '')
                if prompt_content and not prompt_content.startswith('오류:'):
                    st.code(prompt_content, language='text')
                else:
                    st.error("프롬프트 내용을 표시할 수 없습니다.")

def display_results(result: Dict[str, Any]):
    """결과 표시"""
    if not result:
        st.error("추출 결과가 없습니다.")
        return
    

    
    # success 키가 없어도 결과를 표시하도록 수정
    if result.get('success') == False:
        st.warning("API에서 처리 실패를 보고했지만 결과를 확인해보겠습니다.")

    # 탭으로 결과 구분
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 추출 정보", "🔍 추출 JSON", "🔗 DAG 이미지", "🔍 사용된 프롬프트", "📋 메타데이터"])
    
    with tab1:
        st.subheader("추출 정보")
        
        # 추출된 데이터를 표 형태로 표시 (API 응답 구조에 맞게 수정)
        extracted_data = None
        
        # API 응답 구조에 따라 추출된 데이터 찾기
        if 'result' in result:
            # API 응답에서 result 필드가 실제 추출 결과
            extracted_data = result['result']
        # 하위 호환성을 위한 다른 키 확인
        elif 'extracted_data' in result:
            extracted_data = result['extracted_data']
        elif 'ext_result' in result:
            extracted_data = result['ext_result']
        
        if extracted_data:
            # 딕셔너리인 경우
            if isinstance(extracted_data, dict):
                # 카테고리 표시 순서 정의
                preferred_order = ['title', 'purpose', 'product', 'channel', 'pgm', 'entity_dag']
                
                # 선호하는 순서대로 키 정렬
                def get_ordered_keys(data_dict):
                    ordered_keys = []
                    remaining_keys = list(data_dict.keys())
                    
                    # 1. 선호하는 순서대로 키들을 먼저 추가
                    for preferred_key in preferred_order:
                        for key in remaining_keys[:]:  # 복사본으로 순회
                            if key.lower() == preferred_key:
                                ordered_keys.append(key)
                                remaining_keys.remove(key)
                                break
                    
                    # 2. 남은 키들을 알파벳 순서로 뒤에 추가 (새로운 키들 대응)
                    remaining_keys.sort()
                    ordered_keys.extend(remaining_keys)
                    
                    return ordered_keys
                
                # 정렬된 순서로 카테고리 표시
                ordered_categories = get_ordered_keys(extracted_data)
                
                # 각 카테고리별로 데이터 표시
                for category in ordered_categories:
                    items = extracted_data[category]
                    if items:  # 데이터가 있는 경우에만 표시
                        # 카테고리별 아이콘 설정
                        category_icons = {
                            'channel': '📱',
                            'offer': '🎁', 
                            'product': '📦',
                            'entity': '🏷️',
                            'title': '📝',
                            'price': '💰',
                            'date': '📅',
                            'contact': '📞',
                            'purpose': '🎯',
                            'produdt': '📦',  # 오타 버전
                            'pgm': '⚙️',
                            'entity_dag': '🔗'
                        }
                        icon = category_icons.get(category.lower(), '📊')
                        st.markdown(f"### {icon} {category.upper()}")
                        
                        if isinstance(items, list) and len(items) > 0:
                            # 모든 리스트 항목을 DataFrame으로 표시 시도
                            try:
                                # 각 항목이 딕셔너리인지 확인
                                if all(isinstance(item, dict) for item in items):
                                    # 모든 항목의 값이 리스트가 아닌 스칼라 값인지 확인
                                    flattened_items = []
                                    for item in items:
                                        flattened_item = {}
                                        for key, value in item.items():
                                            if isinstance(value, list):
                                                # 리스트인 경우 처리
                                                if len(value) == 0:
                                                    flattened_item[key] = ""
                                                elif len(value) == 1:
                                                    # 단일 항목인 경우, 중첩 구조 확인
                                                    if isinstance(value[0], dict):
                                                        # 딕셔너리 리스트인 경우 JSON 문자열로 변환
                                                        flattened_item[key] = str(value[0])
                                                    else:
                                                        flattened_item[key] = str(value[0])
                                                else:
                                                    # 여러 항목인 경우
                                                    if all(isinstance(v, dict) for v in value):
                                                        # 딕셔너리 리스트인 경우 요약 정보 생성
                                                        summary = []
                                                        for v in value:
                                                            if 'item_nm' in v:
                                                                summary.append(v['item_nm'])
                                                            else:
                                                                summary.append(str(v))
                                                        flattened_item[key] = ', '.join(summary)
                                                    else:
                                                        flattened_item[key] = ', '.join(map(str, value))
                                            else:
                                                # 스칼라 값은 그대로 사용
                                                flattened_item[key] = value
                                        flattened_items.append(flattened_item)
                                    
                                    df = pd.DataFrame(flattened_items)
                                    
                                    # Product 항목의 컬럼 순서 조정
                                    if category.lower() == 'product':
                                        desired_columns = ['item_name_in_msg', 'expected_action', 'item_in_voca']
                                        # 지정된 컬럼들이 존재하는지 확인하고 순서 조정
                                        available_columns = [col for col in desired_columns if col in df.columns]
                                        remaining_columns = [col for col in df.columns if col not in desired_columns]
                                        # 새로운 컬럼 순서: 지정된 순서 + 나머지 컬럼들
                                        new_column_order = available_columns + remaining_columns
                                        df = df[new_column_order]
                                    
                                    st.dataframe(df)
                                else:
                                    # 딕셔너리가 아닌 항목들이 있으면 단순 값들을 DataFrame으로 변환 시도
                                    simple_items = []
                                    for i, item in enumerate(items):
                                        if isinstance(item, (str, int, float)):
                                            # 특정 카테고리는 항목 번호 없이 내용만 표시
                                            if category.lower() in ['entity_dag', 'purpose', 'title']:
                                                simple_items.append({"내용": str(item)})
                                            else:
                                                simple_items.append({"항목": i+1, "내용": str(item)})
                                        else:
                                            if category.lower() in ['entity_dag', 'purpose', 'title']:
                                                simple_items.append({"내용": str(item)})
                                            else:
                                                simple_items.append({"항목": i+1, "내용": str(item)})
                                    
                                    df = pd.DataFrame(simple_items)
                                    st.dataframe(df)
                            except Exception as e:
                                # DataFrame 변환 실패 시 개별 항목으로 표시
                                st.info(f"테이블 형태로 표시할 수 없어 개별 항목으로 표시합니다.")
                                for i, item in enumerate(items, 1):
                                    st.markdown(f"**항목 {i}:**")
                                    if isinstance(item, dict):
                                        for key, value in item.items():
                                            if isinstance(value, list):
                                                st.write(f"**{key}**: {', '.join(map(str, value))}")
                                            else:
                                                st.write(f"**{key}**: {value}")
                                    else:
                                        st.write(item)
                                    if i < len(items):
                                        st.divider()
                        else:
                            # 단일 값이나 기타 형태도 DataFrame 형태로 표시
                            try:
                                if isinstance(items, dict):
                                    # 딕셔너리인 경우 키-값 쌍을 DataFrame으로 변환
                                    dict_items = []
                                    for key, value in items.items():
                                        if isinstance(value, list):
                                            dict_items.append({"속성": key, "값": ', '.join(map(str, value))})
                                        else:
                                            dict_items.append({"속성": key, "값": str(value)})
                                    df = pd.DataFrame(dict_items)
                                    st.dataframe(df)
                                else:
                                    # 단일 값을 DataFrame으로 표시
                                    if category.lower() in ['entity_dag', 'purpose', 'title']:
                                        # 특정 카테고리는 항목 번호 없이 내용만 표시
                                        single_item = [{"내용": str(items)}]
                                    else:
                                        single_item = [{"항목": 1, "내용": str(items)}]
                                    df = pd.DataFrame(single_item)
                                    st.dataframe(df)
                            except Exception as e:
                                # DataFrame 변환 실패 시 기본 표시
                                if isinstance(items, dict):
                                    for key, value in items.items():
                                        if isinstance(value, list):
                                            st.write(f"**{key}**: {', '.join(map(str, value))}")
                                        else:
                                            st.write(f"**{key}**: {value}")
                                else:
                                    st.write(f"**{category}**: {str(items)}")
                
                # 전체 데이터 요약
                st.markdown("### 📋 추출 요약")
                total_items = sum(len(items) if isinstance(items, list) else 1 for items in extracted_data.values() if items)
                st.metric("총 추출된 항목 수", total_items)
                
                # 카테고리별 개수
                col1, col2, col3 = st.columns(3)
                for i, (category, items) in enumerate(extracted_data.items()):
                    if items:
                        count = len(items) if isinstance(items, list) else 1
                        with [col1, col2, col3][i % 3]:
                            st.metric(f"{category}", count)
            
            # 리스트인 경우
            elif isinstance(extracted_data, list):
                st.markdown("### 📋 추출된 데이터 (리스트)")
                for i, item in enumerate(extracted_data):
                    st.markdown(f"**항목 {i+1}:**")
                    st.json(item)
        else:
            st.info("추출된 데이터가 응답에 포함되지 않았습니다.")
            
            # 다른 가능한 키들 확인
            for key, value in result.items():
                if key not in ['success', 'metadata', 'dag_image_url']:
                    st.markdown(f"### 📊 {key.upper()}")
                    if isinstance(value, (dict, list)):
                        st.json(value)
                    else:
                        st.write(value)
    
    with tab2:
        st.subheader("추출 JSON")
        st.json(result)
    
    with tab3:
        st.subheader("DAG 이미지")
        
        # DAG 관련 정보를 다양한 키에서 찾기
        dag_found = False
        dag_info_keys = ['dag_image_url', 'dag_url', 'image_url', 'dag_image', 'dag_path']
        
        # 1. API 응답에서 DAG URL 찾기
        for key in dag_info_keys:
            if key in result and result[key]:
                dag_found = True
                try:
                    dag_url = result[key]
                    
                    # URL이 '/'로 시작하지 않으면 추가
                    if not dag_url.startswith('/'):
                        dag_url = '/' + dag_url
                    
                    full_dag_url = f"{DEMO_API_BASE_URL}{dag_url}"
                    
                    # DAG 이미지 요청
                    dag_response = requests.get(full_dag_url, timeout=10)
                    
                    if dag_response.status_code == 200:
                        st.image(dag_response.content, caption="오퍼 관계 DAG")
                        break  # 성공하면 루프 종료
                    else:
                        st.warning(f"DAG 이미지 응답 오류: {dag_response.status_code}")
                        
                except Exception as e:
                    st.error(f"DAG 이미지 로딩 오류 ({key}): {e}")
                    continue
        
        # 2. 현재 메시지에 해당하는 DAG 이미지 찾기 (메시지 해시 기반)
        if not dag_found and 'result' in st.session_state:
            # 현재 메시지 가져오기 (세션에서)
            current_message = st.session_state.get('current_message', '')
            if current_message:
                try:
                    import hashlib
                    from pathlib import Path
                    
                    message_hash = hashlib.sha256(current_message.encode('utf-8')).hexdigest()
                    expected_filename = f"dag_{message_hash}.png"
                    
                    # 1. 먼저 로컬 파일 시스템에서 확인 (현재 디렉토리 기준으로 우선 확인)
                    possible_dag_paths = [
                        Path(__file__).parent / "dag_images" / expected_filename,
                        Path("dag_images") / expected_filename,
                        Path.cwd() / "dag_images" / expected_filename,
                        Path.cwd() / "mms_extractor_unified" / "dag_images" / expected_filename,
                        Path.cwd() / "mms_extractor_exp" / "dag_images" / expected_filename
                    ]
                    
                    local_file_found = False
                    for dag_path in possible_dag_paths:
                        if dag_path.exists():
                            try:
                                st.image(str(dag_path), caption=f"메시지별 DAG 이미지 ({expected_filename})")
                                dag_found = True
                                local_file_found = True
                                break
                            except Exception as local_error:
                                continue
                    
                    # 2. 로컬에서 찾지 못한 경우 Demo Server를 통해 시도
                    if not local_file_found:
                        try:
                            specific_dag_url = f"{DEMO_API_BASE_URL}/dag_images/{expected_filename}"
                            dag_response = requests.get(specific_dag_url, timeout=5)
                            
                            if dag_response.status_code == 200:
                                # Content-Type 확인
                                content_type = dag_response.headers.get('Content-Type', '')
                                
                                if 'image' in content_type:
                                    st.image(dag_response.content, caption=f"메시지별 DAG 이미지 ({expected_filename})")
                                    dag_found = True
                                else:
                                    st.warning(f"⚠️ 이미지가 아닌 응답: {content_type}")
                        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_error:
                            pass  # 조용히 로컬 파일로 대체
                        except Exception as e:
                            pass  # 기타 오류는 조용히 처리
                        
                except Exception as e:
                    pass  # 오류 메시지 숨김
        
        # 메타데이터에서 DAG 관련 정보 확인
        if not dag_found and 'metadata' in result:
            metadata = result['metadata']
            if metadata.get('extract_entity_dag'):
                pass  # 메시지 없이 바로 로컬 파일 검색으로 진행
                
                # DAG 이미지 디렉토리에서 최신 이미지 찾기 시도

                try:
                    # 1. 직접 DAG 이미지 파일 목록 확인 (API 우회)
                    import os
                    from pathlib import Path
                    
                    # DAG 이미지 디렉토리 경로 (현재 실행 위치 기준으로 우선 확인)
                    current_dir = Path.cwd()
                    
                    # 다양한 가능한 경로 시도 (현재 위치 우선)
                    possible_paths = [
                        Path(__file__).parent / "dag_images",  # 스크립트와 같은 디렉토리 (최우선)
                        Path("dag_images"),  # 현재 디렉토리
                        current_dir / "dag_images",
                        current_dir / "mms_extractor_exp" / "dag_images",
                        current_dir / "mms_extractor_unified" / "dag_images"
                    ]
                    
                    for i, path in enumerate(possible_paths):
                        exists = path.exists()
                        if exists:
                            dag_images_dir = path
                            break
                    
                    if dag_images_dir.exists():
                        # DAG 이미지 파일 목록 가져오기
                        dag_files = list(dag_images_dir.glob("dag_*.png"))
                        
                        if dag_files:
                            # 가장 최근 파일 선택 (수정 시간 기준)
                            latest_file = max(dag_files, key=lambda x: x.stat().st_mtime)
                            
                            # 우선 로컬 파일 직접 읽기 시도
                            try:
                                if latest_file.exists() and latest_file.is_file():
                                    st.image(str(latest_file), caption=f"DAG 이미지 (로컬) - {latest_file.name}")
                                    dag_found = True
                                    pass  # 성공 메시지 제거
                                else:
                                    # 로컬 파일이 없으면 Demo Server를 통해 시도
                                    latest_dag_url = f"{DEMO_API_BASE_URL}/dag_images/{latest_file.name}"
                                    
                                    image_response = requests.get(latest_dag_url, timeout=5)
                                    if image_response.status_code == 200:
                                        st.image(image_response.content, caption=f"DAG 이미지 ({latest_file.name})")
                                        dag_found = True
                                        pass  # 성공 메시지 제거
                                    else:
                                        st.warning(f"⚠️ DAG 이미지 로딩 실패: {image_response.status_code}")
                            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_error:
                                pass  # 조용히 처리
                            except Exception as local_error:
                                st.warning(f"⚠️ 이미지 로딩 중 오류: {local_error}")
                        else:
                            st.info("DAG 이미지 파일이 생성되지 않았습니다.")
                    else:
                        st.warning("DAG 이미지 디렉토리를 찾을 수 없습니다.")
                        
                        # 대안: API를 통한 검색 시도
                        try:
                            dag_list_response = requests.get(f"{DEMO_API_BASE_URL}/api/dag-images", timeout=10)

                            
                            if dag_list_response.status_code == 200 and dag_list_response.text.strip():
                                dag_list_data = dag_list_response.json()
                                if dag_list_data.get('images'):
                                    latest_image = dag_list_data['images'][0]
                                    latest_dag_url = f"{DEMO_API_BASE_URL}{latest_image['url']}"
                                    
                                    image_response = requests.get(latest_dag_url, timeout=10)
                                    if image_response.status_code == 200:
                                        st.image(image_response.content, caption=f"DAG 이미지 ({latest_image['filename']})")
                                        dag_found = True
                        except Exception as api_error:
                            pass  # 오류 메시지 숨김
                            
                except Exception as e:
                    pass  # 오류 메시지 숨김
            else:
                st.info("DAG 추출이 비활성화되어 있습니다.")
        
        if not dag_found:
            st.info("DAG 그래프를 찾을 수 없습니다.")
            
            # 추가 진단 정보
            with st.expander("🔧 진단 정보 및 해결 방법"):
                st.markdown("### 확인 사항:")
                st.markdown("""
                - ✅ DAG 추출 옵션이 활성화되어 있는지 확인
                - ✅ Demo Server (포트 8082)가 실행 중인지 확인  
                - ✅ DAG 이미지 생성에 시간이 걸릴 수 있습니다
                """)
                
                # Demo Server 연결 테스트
                st.markdown("### Demo Server 연결 테스트:")
                try:
                    demo_health = requests.get(f"{DEMO_API_BASE_URL}/api/dag-images", timeout=5)
                    if demo_health.status_code == 200:
                        st.success("✅ Demo Server 연결 성공")
                        data = demo_health.json()
                        st.write(f"📊 DAG 이미지 개수: {len(data.get('images', []))}")
                    else:
                        st.error(f"❌ Demo Server 응답 오류: {demo_health.status_code}")
                except Exception as e:
                    st.error(f"❌ Demo Server 연결 실패: {e}")
                
                # 수동 DAG 이미지 확인
                st.markdown("### 수동 DAG 이미지 확인:")
                if st.button("🔍 DAG 이미지 디렉토리 확인"):
                    try:
                        dag_response = requests.get(f"{DEMO_API_BASE_URL}/api/dag-images")
                        if dag_response.status_code == 200:
                            dag_data = dag_response.json()
                            if dag_data.get('images'):
                                st.success(f"✅ {len(dag_data['images'])}개의 DAG 이미지 발견")
                                for img in dag_data['images'][:3]:  # 최대 3개만 표시
                                    st.write(f"- {img['filename']} (크기: {img['size']} bytes)")
                            else:
                                st.warning("⚠️ DAG 이미지가 없습니다.")
                        else:
                            st.error(f"❌ API 오류: {dag_response.status_code}")
                    except Exception as e:
                        st.error(f"❌ 확인 중 오류: {e}")
                
                # 디버깅: 모든 응답 키 표시
                st.markdown("### API 응답 분석:")
                dag_related_keys = []
                for key, value in result.items():
                    if 'dag' in key.lower() or 'image' in key.lower():
                        dag_related_keys.append(f"{key}: {value}")
                
                if dag_related_keys:
                    st.write("DAG/이미지 관련 키:")
                    for key_info in dag_related_keys:
                        st.write(f"- {key_info}")
                else:
                    st.write("DAG/이미지 관련 키가 응답에 없습니다.")
    
    with tab4:
        st.subheader("실제 사용된 LLM 프롬프트")
        
        # 추출 결과에서 프롬프트 정보 가져오기
        if 'result' in st.session_state:
            # 동일한 설정으로 프롬프트 가져오기
            current_message = st.session_state.get('current_message', '')
            if current_message:
                # 현재 결과의 메타데이터에서 설정 정보 가져오기
                metadata = result.get('metadata', {})
                
                prompt_data = {
                    "message": current_message,
                    "llm_model": metadata.get('llm_model', 'ax'),
                    "offer_info_data_src": metadata.get('offer_info_data_src', 'local'),
                    "product_info_extraction_mode": metadata.get('product_info_extraction_mode', 'llm'),
                    "entity_matching_mode": metadata.get('entity_matching_mode', 'logic'),
                    "extract_entity_dag": metadata.get('extract_entity_dag', False)
                }
                
                # 세션에 저장된 프롬프트가 있는지 확인
                if 'extraction_prompts' in st.session_state and st.session_state['extraction_prompts']:
                    prompts_result = st.session_state['extraction_prompts']
                    st.info("위 추출 과정에서 실제로 LLM에 전송된 프롬프트들입니다.")
                    
                    # 디버깅 정보 추가
                    with st.expander("🔧 프롬프트 디버깅 정보"):
                        st.write(f"prompts_result 타입: {type(prompts_result)}")
                        st.write(f"prompts_result 키들: {list(prompts_result.keys()) if isinstance(prompts_result, dict) else 'dict가 아님'}")
                        if isinstance(prompts_result, dict):
                            st.write(f"success 값: {prompts_result.get('success')}")
                            st.write(f"prompts 키 존재: {'prompts' in prompts_result}")
                            if 'prompts' in prompts_result:
                                st.write(f"prompts 내용: {prompts_result['prompts']}")
                    
                    if prompts_result.get('success'):
                        prompts = prompts_result.get('prompts', {})
                        settings = prompts_result.get('settings', {})
                        
                        # 설정 정보 표시
                        with st.expander("⚙️ 추출 시 사용된 설정"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**LLM 모델**: {settings.get('llm_model', 'N/A')}")
                                st.write(f"**데이터 소스**: {settings.get('offer_info_data_src', 'N/A')}")
                                st.write(f"**상품 추출 모드**: {settings.get('product_info_extraction_mode', 'N/A')}")
                            with col2:
                                st.write(f"**엔티티 매칭 모드**: {settings.get('entity_matching_mode', 'N/A')}")
                                st.write(f"**DAG 추출**: {'활성화' if settings.get('extract_entity_dag', False) else '비활성화'}")
                        
                        # 프롬프트 표시 순서 정의
                        prompt_order = [
                            'main_extraction_prompt',  # 메인 정보 추출
                            'entity_extraction_prompt',  # 엔티티 추출
                            'dag_extraction_prompt'  # DAG 관계 추출
                        ]
                        
                        # 정의된 순서대로 프롬프트 표시
                        for prompt_key in prompt_order:
                            if prompt_key in prompts:
                                prompt_info = prompts[prompt_key]
                                with st.expander(f"📝 {prompt_info.get('title', prompt_key)}"):
                                    st.write(f"**설명**: {prompt_info.get('description', '설명 없음')}")
                                    st.write(f"**길이**: {prompt_info.get('length', 0):,} 문자")
                                    
                                    prompt_content = prompt_info.get('content', '')
                                    if prompt_content and not prompt_content.startswith('오류:'):
                                        st.code(prompt_content, language='text')
                                    else:
                                        st.error("프롬프트 내용을 표시할 수 없습니다.")
                        
                        # 순서에 없는 다른 프롬프트들도 표시
                        for prompt_key, prompt_info in prompts.items():
                            if prompt_key not in prompt_order:
                                with st.expander(f"📝 {prompt_info.get('title', prompt_key)}"):
                                    st.write(f"**설명**: {prompt_info.get('description', '설명 없음')}")
                                    st.write(f"**길이**: {prompt_info.get('length', 0):,} 문자")
                                    
                                    prompt_content = prompt_info.get('content', '')
                                    if prompt_content and not prompt_content.startswith('오류:'):
                                        st.code(prompt_content, language='text')
                                    else:
                                        st.error("프롬프트 내용을 표시할 수 없습니다.")
                    else:
                        st.error("프롬프트 데이터를 가져올 수 없습니다.")
                        if 'error' in prompts_result:
                            st.error(f"오류: {prompts_result['error']}")
                else:
                    st.warning("⚠️ 프롬프트 정보를 가져올 수 없었습니다.")
                    st.info("정보 추출을 다시 실행해보세요.")
            else:
                st.warning("메시지 정보가 없습니다.")
        else:
            st.info("추출 결과가 없습니다. 먼저 정보 추출을 실행해주세요.")

    with tab5:
        st.subheader("메타데이터")
        
        if 'metadata' in result:
            metadata = result['metadata']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("처리 시간", f"{metadata.get('processing_time_seconds', 'N/A')}초")
                st.metric("LLM 모델", metadata.get('llm_model', 'N/A'))
                st.metric("데이터 소스", metadata.get('offer_info_data_src', 'N/A'))
            
            with col2:
                st.metric("상품 추출 모드", metadata.get('product_info_extraction_mode', 'N/A'))
                st.metric("엔티티 매칭 모드", metadata.get('entity_matching_mode', 'N/A'))
                st.metric("DAG 추출", "활성화" if metadata.get('extract_entity_dag') else "비활성화")
        else:
            st.info("메타데이터가 응답에 포함되지 않았습니다.")
    


def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📱 MMS Extractor API Demo</h1>
        <p>MMS 메시지에서 구조화된 정보를 추출하는 AI 서비스</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API 상태 확인
    api_status = check_api_status()
    status_text = "🟢 API 서버 연결됨" if api_status else "🔴 API 서버 오프라인"
    status_class = "status-online" if api_status else "status-offline"
    
    st.markdown(f'<p class="{status_class}">🔍 API 상태: {status_text}</p>', unsafe_allow_html=True)
    
    # 현재 포트 설정 표시
    st.info(f"📡 **현재 설정**: API 서버 포트 {args.api_port}, Demo 서버 포트 {args.demo_port}")
    
    # 메인 탭 생성 (단일 처리 vs 배치 처리)
    main_tab1, main_tab2 = st.tabs(["📄 단일 메시지 처리", "📋 배치 처리"])
    
    with main_tab1:
        # 단일 처리 UI
        display_single_processing_ui(api_status, args)
    
    with main_tab2:
        # 배치 처리 UI
        display_batch_processing_ui(api_status, args)

def display_single_processing_ui(api_status: bool, args):
    """단일 메시지 처리 UI"""
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 단일 메시지 설정")
        
        # LLM 모델 선택
        llm_model = st.selectbox(
            "LLM 모델",
            ["ax", "gemma", "claude", "gemini"],
            format_func=lambda x: {
                "ax": "A.X (SKT)",
                "gemma": "Gemma",
                "claude": "Claude", 
                "gemini": "Gemini"
            }[x]
        )
        
        # 데이터 소스
        data_source = st.selectbox(
            "데이터 소스",
            ["local", "db"],
            format_func=lambda x: "Local (CSV)" if x == "local" else "Database"
        )
        
        # 상품 추출 모드
        product_mode = st.selectbox(
            "상품 추출 모드",
            ["nlp", "llm", "rag"],
            format_func=lambda x: {
                "nlp": "NLP (형태소 분석)",
                "llm": "LLM 기반",
                "rag": "RAG (검색증강)"
            }[x]
        )
        
        # 엔티티 매칭 모드
        entity_mode = st.selectbox(
            "개체명 추출 모드", 
            ["logic", "llm"],
            format_func=lambda x: "통합 LLM 기반" if x == "logic" else "분리 LLM 기반"
        )
        
        # DAG 추출 옵션
        extract_dag = st.checkbox("오퍼 관계 DAG 추출", value=True)
        
    # 메인 컨텐츠 (메시지 입력 부분을 줄이고 추출 결과 부분을 키움)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("💡 샘플 메시지")
        
        # 샘플 메시지 선택
        for i, sample in enumerate(SAMPLE_MESSAGES):
            if st.button(sample["title"], key=f"sample_{i}"):
                st.session_state['message_input'] = sample["content"]
                st.rerun()
        
        # 메시지 입력
        st.subheader("📝 메시지 입력")
        
        # 세션 상태에서 메시지 가져오기 (key와 동일한 이름 사용)
        default_message = st.session_state.get('message_input', '')
        
        message = st.text_area(
            "MMS 메시지 내용",
            value=default_message,
            height=300,
            placeholder="추출하고 싶은 MMS 메시지를 입력하세요...",
            key="message_input"
        )
        
        # 참고: 프롬프트는 정보 추출 실행 후 결과와 함께 표시됩니다
        
        # 추출 실행 버튼
        st.write(f"🔍 API 상태: {api_status}")
        st.write(f"📝 메시지 길이: {len(message.strip()) if message else 0}")
        
        if st.button("🚀 정보 추출 실행", type="primary", disabled=not api_status):
            st.write("🎯 버튼이 클릭되었습니다!")
            
            if not message.strip():
                st.error("메시지를 입력해주세요.")
            else:
                st.write("📋 API 호출 데이터 준비 중...")
                
                # API 호출 데이터 준비
                api_data = {
                    "message": message,
                    "llm_model": llm_model,
                    "offer_info_data_src": data_source,
                    "product_info_extraction_mode": product_mode,
                    "entity_matching_mode": entity_mode,
                    "extract_entity_dag": extract_dag
                }
                
                st.write("🔄 API 호출 시작...")
                
                # 진행 상황 표시
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text("🚀 API 서버로 요청 전송 중...")
                progress_bar.progress(10)
                
                # 로딩 상태 표시
                with st.spinner("정보를 추출하는 중입니다... (최대 2분 소요)"):
                    progress_text.text("🤖 AI가 메시지를 분석하고 있습니다...")
                    progress_bar.progress(30)
                    
                    result = call_extraction_api(api_data)
                    
                    if result:
                        progress_text.text("✅ 처리 완료!")
                        progress_bar.progress(100)
                    else:
                        progress_text.text("❌ 처리 실패")
                        progress_bar.progress(0)
                
                if result:
                    st.session_state['result'] = result
                    # 추출 결과에 프롬프트가 포함되어 있으면 사용, 없으면 별도 API 호출
                    if 'prompts' in result and result['prompts'].get('success'):
                        st.session_state['extraction_prompts'] = result['prompts']
                        st.info("✅ 프롬프트가 추출 결과와 함께 반환되었습니다.")
                    else:
                        # 기존 방식: 별도 프롬프트 API 호출
                        prompts_result = call_prompts_api(api_data)
                        st.session_state['extraction_prompts'] = prompts_result
                        st.info("✅ 프롬프트를 별도로 가져왔습니다.")
                    
                    st.session_state['current_message'] = message  # 현재 메시지 저장
                    
                    st.rerun()  # 페이지 새로고침으로 결과 표시
                else:
                    st.error("❌ 추출 중 오류가 발생했습니다.")
    
    with col2:
        st.subheader("📊 작업 결과")
        
        # 결과 표시
        if 'result' in st.session_state:
            display_results(st.session_state['result'])
        else:
            st.info("메시지를 입력하고 '정보 추출 실행' 버튼을 클릭하세요.")

def display_batch_processing_ui(api_status: bool, args):
    """배치 메시지 처리 UI"""
    st.header("📋 배치 메시지 처리")
    st.info("여러 메시지를 한 번에 처리할 수 있습니다. 각 메시지는 빈 줄로 구분해주세요.")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 배치 처리 설정")
        
        # LLM 모델 선택
        batch_llm_model = st.selectbox(
            "LLM 모델 (배치)",
            ["ax", "gemma", "claude", "gemini"],
            format_func=lambda x: {
                "ax": "A.X (SKT)",
                "gemma": "Gemma",
                "claude": "Claude", 
                "gemini": "Gemini"
            }[x],
            key="batch_llm_model"
        )
        
        # 데이터 소스
        batch_data_source = st.selectbox(
            "데이터 소스 (배치)",
            ["local", "db"],
            format_func=lambda x: "Local (CSV)" if x == "local" else "Database",
            key="batch_data_source"
        )
        
        # 상품 추출 모드
        batch_product_mode = st.selectbox(
            "상품 추출 모드 (배치)",
            ["nlp", "llm", "rag"],
            format_func=lambda x: {
                "nlp": "NLP (형태소 분석)",
                "llm": "LLM 기반",
                "rag": "RAG (검색증강)"
            }[x],
            key="batch_product_mode"
        )
        
        # 엔티티 매칭 모드
        batch_entity_mode = st.selectbox(
            "엔티티 매칭 모드 (배치)",
            ["logic", "llm"],
            format_func=lambda x: "통합 LLM 기반" if x == "logic" else "분리 LLM 기반",
            key="batch_entity_mode"
        )
        
        # 최대 워커 수
        max_workers = st.selectbox(
            "최대 워커 수",
            [2, 4, 8, 16],
            index=0,
            help="워커 수가 많을수록 빠르지만 시스템 리소스를 더 많이 사용합니다."
        )
        
        # DAG 추출 옵션
        batch_extract_dag = st.checkbox(
            "🔗 오퍼 관계 DAG 추출 (배치)",
            help="엔티티 간 관계를 DAG 형태로 추출하여 시각화합니다.",
            key="batch_extract_dag"
        )
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 배치 메시지 입력")
        
        # 샘플 메시지 로드 버튼
        if st.button("📋 샘플 메시지 로드", key="load_batch_samples"):
            sample_text = "\n\n".join([msg["content"] for msg in SAMPLE_MESSAGES])
            st.session_state['batch_messages'] = sample_text
        
        # 배치 메시지 입력
        batch_messages = st.text_area(
            "배치 메시지 입력 *",
            value=st.session_state.get('batch_messages', ''),
            height=300,
            placeholder="여러 메시지를 입력하세요. 각 메시지는 빈 줄로 구분합니다.\n\n예시:\n[SKT] 첫 번째 메시지\n\n[KT] 두 번째 메시지\n\n[LG U+] 세 번째 메시지",
            help="각 메시지는 빈 줄로 구분해주세요. 최대 100개 메시지까지 처리 가능합니다.",
            key="batch_messages_input"
        )
        
        # 메시지 개수 표시
        if batch_messages:
            messages_list = [msg.strip() for msg in batch_messages.split('\n\n') if msg.strip()]
            st.info(f"📊 입력된 메시지 개수: {len(messages_list)}개")
            
            if len(messages_list) > 100:
                st.warning("⚠️ 메시지가 100개를 초과합니다. 처리 시간이 매우 오래 걸릴 수 있습니다.")
        
        # 배치 처리 실행 버튼
        if st.button("🚀 배치 처리 실행", type="primary", disabled=not api_status, key="batch_submit"):
            if not batch_messages:
                st.error("처리할 메시지를 입력해주세요.")
            else:
                messages_list = [msg.strip() for msg in batch_messages.split('\n\n') if msg.strip()]
                
                with st.spinner(f"배치 처리 중... ({len(messages_list)}개 메시지)"):
                    # 진행률 표시
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # API 호출
                    result = call_batch_api(
                        messages_list,
                        batch_llm_model,
                        batch_data_source,
                        batch_product_mode,
                        batch_entity_mode,
                        batch_extract_dag,
                        max_workers
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 배치 처리 완료!")
                    
                    if result:
                        st.session_state['batch_result'] = result
                        st.session_state['batch_messages_processed'] = messages_list
                        st.success(f"✅ {len(messages_list)}개 메시지 처리가 완료되었습니다!")
                        st.rerun()
                    else:
                        st.error("❌ 배치 처리 중 오류가 발생했습니다.")
    
    with col2:
        st.subheader("📊 배치 처리 결과")
        
        # 배치 결과 표시
        if 'batch_result' in st.session_state:
            display_batch_results(st.session_state['batch_result'])
        else:
            st.info("배치 메시지를 입력하고 '배치 처리 실행' 버튼을 클릭하세요.")

def call_batch_api(messages: list, llm_model: str, data_source: str, product_mode: str, 
                   entity_mode: str, extract_dag: bool, max_workers: int) -> Optional[Dict[str, Any]]:
    """배치 API 호출"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch",
            json={
                "messages": messages,
                "llm_model": llm_model,
                "offer_info_data_src": data_source,
                "product_info_extraction_mode": product_mode,
                "entity_matching_mode": entity_mode,
                "extract_entity_dag": extract_dag,
                "max_workers": max_workers
            },
            timeout=300  # 5분 타임아웃
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API 호출 실패: {response.status_code}")
            st.error(f"오류 메시지: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"API 호출 오류: {str(e)}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        return None

def display_batch_results(result: Dict[str, Any]):
    """배치 결과 표시"""
    if not result:
        st.error("배치 처리 결과가 없습니다.")
        return
    
    if not result.get('success', True):
        st.error(f"배치 처리 실패: {result.get('error', '알 수 없는 오류')}")
        return
    
    # 요약 정보
    metadata = result.get('metadata', {})
    results_list = result.get('results', [])
    
    total_count = len(results_list)
    success_count = sum(1 for r in results_list if r.get('success', False))
    failure_count = total_count - success_count
    processing_time = metadata.get('processing_time_seconds', 0)
    
    # 요약 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 메시지", total_count)
    
    with col2:
        st.metric("성공", success_count, delta=None, delta_color="normal")
    
    with col3:
        st.metric("실패", failure_count, delta=None, delta_color="inverse")
    
    with col4:
        st.metric("처리 시간", f"{processing_time:.1f}초")
    
    # 추가 메타데이터
    st.subheader("📋 처리 설정")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**LLM 모델**: {metadata.get('llm_model', 'N/A')}")
        st.info(f"**워커 수**: {metadata.get('max_workers', 'N/A')}")
    
    with col2:
        st.info(f"**데이터 소스**: {metadata.get('offer_info_data_src', 'N/A')}")
        st.info(f"**DAG 추출**: {'ON' if metadata.get('extract_entity_dag', False) else 'OFF'}")
    
    with col3:
        st.info(f"**상품 추출**: {metadata.get('product_info_extraction_mode', 'N/A')}")
        st.info(f"**엔티티 매칭**: {metadata.get('entity_matching_mode', 'N/A')}")
    
    # 개별 결과 표시
    st.subheader("📄 개별 처리 결과")
    
    # 필터링 옵션
    filter_option = st.selectbox(
        "결과 필터",
        ["전체", "성공만", "실패만"],
        key="batch_filter"
    )
    
    filtered_results = results_list
    if filter_option == "성공만":
        filtered_results = [r for r in results_list if r.get('success', False)]
    elif filter_option == "실패만":
        filtered_results = [r for r in results_list if not r.get('success', False)]
    
    # 결과 표시
    for i, item in enumerate(filtered_results):
        with st.expander(f"메시지 {item.get('index', i) + 1}: {'✅ 성공' if item.get('success', False) else '❌ 실패'}"):
            if item.get('success', False):
                # 성공한 경우 - 추출된 정보 표시
                result_data = item.get('result', {})
                
                if result_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result_data.get('title'):
                            st.write(f"**제목**: {result_data['title']}")
                        if result_data.get('purpose'):
                            st.write(f"**목적**: {result_data['purpose']}")
                        if result_data.get('product'):
                            st.write(f"**상품**: {result_data['product']}")
                    
                    with col2:
                        if result_data.get('channel'):
                            st.write(f"**채널**: {result_data['channel']}")
                        if result_data.get('program'):
                            st.write(f"**프로그램**: {result_data['program']}")
                    
                    # DAG 정보가 있으면 표시
                    if result_data.get('entity_dag'):
                        st.write("**엔티티 관계 (DAG):**")
                        dag_items = result_data['entity_dag']
                        if isinstance(dag_items, list):
                            for dag_item in dag_items[:5]:  # 처음 5개만 표시
                                st.write(f"- {dag_item}")
                            if len(dag_items) > 5:
                                st.write(f"... 및 {len(dag_items) - 5}개 더")
                
                # JSON 데이터 표시 (접을 수 있는 형태)
                with st.expander("🔍 상세 JSON 데이터"):
                    st.json(result_data)
            
            else:
                # 실패한 경우 - 오류 정보 표시
                st.error(f"오류: {item.get('error', '알 수 없는 오류')}")

if __name__ == "__main__":
    main()