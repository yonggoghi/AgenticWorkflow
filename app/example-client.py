import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API endpoint
url = "http://localhost:5001/api/extract"
headers = {"Content-Type": "application/json"}

# Example data
data = {
    "message_head": "[SK텔레콤] ACT대리점 SKT 신현본점에서 꼭꼭 숨겨뒀던 할인 꿀팁 방출",
    "message_body": """(광고)[SKT] 고객님 안녕하세요._루원이편한세상 하늘채아파트 정문 신현본점에서_통신비 할인 비책 공개합니다!__① 기가인터넷+TV 요금무료_ 5월 한정 프로모션, 하나 원더카드가_ SK인터넷요금 2년간 지원_ (전월 카드 40만원이상 사용기준)__② S23 할인이 중복_ S23은 약정할인과 제휴카드 할인이 중복된다는 사실!_ 갓성비 갤럭시 퀀텀3도 마찬가지__③ SK공식인증매장에서 더 싸다?_ SK매직/ADT캡스와 제휴하여_ 같은 공기청정기/정수기/홈보안이라도 훨씬 저렴하게 이용!__□ 매장으로 문의주세요 (네이버,매장 홈페이지로 방문예약 가능)__◆ 단골등록 이벤트_ 단골고객 모두에게_ 스마트폰 구매 시 나와 지인 같이 할인받는_ "지인찬스 쿠폰" 4만원권을 발송__■ ACT대리점 신현본점_- 주소: 인천 서구 가정로 387, 가동 201호 SK텔레콤_- 연락처: 0507-1404-2560_- 네이버로 보기: http://t-mms.kr/t.do?m=#61&s=19725&a=&u=https://m.place.naver.com/place/1108219331/home_- 매장 위치보기: http://t-mms.kr/t.do?m=#61&s=19726&a=&u=https://m.tworld.co.kr/customer/agentsearch/detail?code=D13546-0297_- 매장 홈페이지: http://t-mms.kr/t.do?m=#61&s=19727&a=&u=https://tworldfriends.co.kr/D135460297__■ 문의 : SKT고객센터(1558,무료)_무료 수신거부 1504_""",
    "model_type": "ax"  # Using Claude 3.7
}

# Make the API request
try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an exception for bad status codes
    
    result = response.json()
    print("API Response:")
    print("------------")
    print(json.dumps(result, indent=4, ensure_ascii=False))
    
except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        print(f"Response body: {e.response.text}")
