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

# 커맨드라인 인자 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='MMS Extractor Streamlit Demo')
    parser.add_argument('--api-port', type=int, default=8000, help='API 서버 포트 (기본값: 8000)')
    parser.add_argument('--demo-port', type=int, default=8082, help='Demo 서버 포트 (기본값: 8082)')
    
    # Streamlit이 실행될 때 추가되는 인자들을 무시
    known_args, unknown_args = parser.parse_known_args()
    return known_args

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
                st.write(f"처리 시간: {metadata.get('processing_time', 'N/A')}초")
                
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

def display_results(result: Dict[str, Any]):
    """결과 표시"""
    if not result:
        st.error("추출 결과가 없습니다.")
        return
    

    
    # success 키가 없어도 결과를 표시하도록 수정
    if result.get('success') == False:
        st.warning("API에서 처리 실패를 보고했지만 결과를 확인해보겠습니다.")

    # 탭으로 결과 구분
    tab1, tab2, tab3, tab4 = st.tabs(["📊 추출 정보", "🔍 추출 JSON", "🔗 DAG 이미지", "📋 메타데이터"])
    
    with tab1:
        st.subheader("추출 정보")
        
        # 추출된 데이터를 표 형태로 표시 (API 응답 구조에 맞게 수정)
        extracted_data = None
        
        # 'result' 키에서 추출된 데이터 찾기
        if 'result' in result:
            extracted_data = result['result']
            
        # 'extracted_data' 키에서도 확인 (하위 호환성)
        elif 'extracted_data' in result:
            extracted_data = result['extracted_data']
        
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
                                                # 리스트인 경우 문자열로 변환
                                                if len(value) == 1:
                                                    flattened_item[key] = value[0]
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
                                    
                                    st.dataframe(df, use_container_width=True)
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
                                    st.dataframe(df, use_container_width=True)
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
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    # 단일 값을 DataFrame으로 표시
                                    if category.lower() in ['entity_dag', 'purpose', 'title']:
                                        # 특정 카테고리는 항목 번호 없이 내용만 표시
                                        single_item = [{"내용": str(items)}]
                                    else:
                                        single_item = [{"항목": 1, "내용": str(items)}]
                                    df = pd.DataFrame(single_item)
                                    st.dataframe(df, use_container_width=True)
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
                        st.image(dag_response.content, caption="오퍼 관계 DAG", use_container_width=True)
                        break  # 성공하면 루프 종료
                    else:
                        st.warning(f"DAG 이미지 응답 오류: {dag_response.status_code}")
                        
                except Exception as e:
                    st.error(f"DAG 이미지 로딩 오류 ({key}): {e}")
                    continue
        
        # 2. 현재 메시지에 해당하는 DAG 이미지 찾기 (메시지 해시 기반)
        if not dag_found and 'extraction_result' in st.session_state:
            # 현재 메시지 가져오기 (세션에서)
            current_message = st.session_state.get('current_message', '')
            if current_message:
                try:
                    import hashlib
                    from pathlib import Path
                    
                    message_hash = hashlib.sha256(current_message.encode('utf-8')).hexdigest()
                    expected_filename = f"dag_{message_hash}.png"
                    
                    # 1. 먼저 로컬 파일 시스템에서 확인
                    possible_dag_paths = [
                        Path.cwd() / "mms_extractor_unified" / "dag_images" / expected_filename,
                        Path("dag_images") / expected_filename,
                        Path(__file__).parent / "dag_images" / expected_filename
                    ]
                    
                    local_file_found = False
                    for dag_path in possible_dag_paths:
                        if dag_path.exists():
                            try:
                                st.image(str(dag_path), caption=f"메시지별 DAG 이미지 ({expected_filename})", use_container_width=True)
                                dag_found = True
                                local_file_found = True
                                break
                            except Exception as local_error:
                                continue
                    
                    # 2. 로컬에서 찾지 못한 경우 Demo Server를 통해 시도
                    if not local_file_found:
                        specific_dag_url = f"{DEMO_API_BASE_URL}/dag_images/{expected_filename}"
                        dag_response = requests.get(specific_dag_url, timeout=10)
                        
                        if dag_response.status_code == 200:
                            # Content-Type 확인
                            content_type = dag_response.headers.get('Content-Type', '')
                            
                            if 'image' in content_type:
                                st.image(dag_response.content, caption=f"메시지별 DAG 이미지 ({expected_filename})", use_container_width=True)
                                dag_found = True
                            else:
                                st.warning(f"⚠️ 이미지가 아닌 응답: {content_type}")
                                st.text(f"응답 내용: {dag_response.text[:200]}")
                        
                except Exception as e:
                    pass  # 오류 메시지 숨김
        
        # 메타데이터에서 DAG 관련 정보 확인
        if not dag_found and 'metadata' in result:
            metadata = result['metadata']
            if metadata.get('extract_entity_dag'):
                st.info("DAG 추출이 활성화되어 있지만 이미지 URL을 찾을 수 없습니다.")
                
                # DAG 이미지 디렉토리에서 최신 이미지 찾기 시도

                try:
                    # 1. 직접 DAG 이미지 파일 목록 확인 (API 우회)
                    import os
                    from pathlib import Path
                    
                    # DAG 이미지 디렉토리 경로 (절대 경로 사용)
                    current_dir = Path.cwd()
                    dag_images_dir = current_dir / "mms_extractor_unified" / "dag_images"
                    
                    # 다양한 가능한 경로 시도
                    possible_paths = [
                        dag_images_dir,
                        Path("dag_images"),  # 현재 디렉토리에서 실행된 경우
                        current_dir / "dag_images",
                        Path(__file__).parent / "dag_images"  # 스크립트와 같은 디렉토리
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
                            
                            # Demo Server를 통해 이미지 로드
                            latest_dag_url = f"{DEMO_API_BASE_URL}/dag_images/{latest_file.name}"
                            
                            image_response = requests.get(latest_dag_url, timeout=10)
                            if image_response.status_code == 200:
                                st.image(image_response.content, caption=f"DAG 이미지 ({latest_file.name})", use_container_width=True)
                                dag_found = True
                                st.success("✅ DAG 이미지를 성공적으로 로드했습니다!")
                            else:
                                st.warning(f"DAG 이미지 로딩 실패: {image_response.status_code}")
                                
                                # 대안: 로컬 파일 직접 읽기
                                try:
                                    st.write(f"📁 로컬 파일 직접 읽기 시도: {latest_file}")
                                    if latest_file.exists() and latest_file.is_file():
                                        st.image(str(latest_file), caption=f"DAG 이미지 (로컬) - {latest_file.name}", use_container_width=True)
                                        dag_found = True
                                        st.success("✅ 로컬 파일에서 DAG 이미지를 로드했습니다!")
                                    else:
                                        st.error(f"로컬 파일이 존재하지 않습니다: {latest_file}")
                                except Exception as local_error:
                                    st.error(f"로컬 파일 읽기 실패: {local_error}")
                                    import traceback
                                    st.text(f"상세 오류: {traceback.format_exc()}")
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
                                        st.image(image_response.content, caption=f"DAG 이미지 ({latest_image['filename']})", use_container_width=True)
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
        st.subheader("메타데이터")
        
        if 'metadata' in result:
            metadata = result['metadata']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("처리 시간", f"{metadata.get('processing_time', 'N/A')}초")
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
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 메시지 입력 및 설정")
        
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
            if st.button(sample["title"], key=f"sample_{i}", use_container_width=True):
                st.session_state['selected_message'] = sample["content"]
                st.rerun()
        
        # 메시지 입력
        st.subheader("📝 메시지 입력")
        
        # 세션 상태에서 메시지 가져오기
        default_message = st.session_state.get('selected_message', '')
        
        message = st.text_area(
            "MMS 메시지 내용",
            value=default_message,
            height=300,
            placeholder="추출하고 싶은 MMS 메시지를 입력하세요...",
            key="message_input"
        )
        
        # 추출 실행 버튼
        st.write(f"🔍 API 상태: {api_status}")
        st.write(f"📝 메시지 길이: {len(message.strip()) if message else 0}")
        
        if st.button("🚀 정보 추출 실행", type="primary", use_container_width=True, disabled=not api_status):
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
                    st.session_state['extraction_result'] = result
                    st.session_state['current_message'] = message  # 현재 메시지 저장
                    st.success("✅ 정보 추출이 완료되었습니다!")
                    st.rerun()  # 페이지 새로고침으로 결과 표시
                else:
                    st.error("❌ 추출 중 오류가 발생했습니다.")
    
    with col2:
        st.subheader("📊 작업 결과")
        
        # 결과 표시
        if 'extraction_result' in st.session_state:
            display_results(st.session_state['extraction_result'])
        else:
            st.info("메시지를 입력하고 '정보 추출 실행' 버튼을 클릭하세요.")

if __name__ == "__main__":
    main()