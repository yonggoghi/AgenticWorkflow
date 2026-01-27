import requests
import json
# Extract information
response = requests.post('http://127.0.0.1:8000/extract', json={
    "message": """
(광고)[SKT] 9월 T day 혜택 안내__2025년 9월 17일(수)_럭키찬스!_아이폰 출시 기념 퀴즈_올리브영 기프트 카드 1만 원 응모 이벤트__▶ 자세히 보기: https://t-mms.kr/t.do?m=#61&s=33670&a=&u=https://bit.ly/467rn3q__■ 문의: SKT 고객센터(1558, 무료)__무료 수신거부 1504

""",
"message_id": "test_message_id",
    "llm_model": "ax",  # CLI와 동일하게 Gemini 사용 (API에서는 'gemini'로 표기)
    "product_info_extraction_mode": "llm",
    "entity_matching_mode": "llm",  # 올바른 파라미터 이름
    "extract_entity_dag": True,
    "result_type": "ext",
    "save_to_mongodb": True,
    
})

# response = requests.post('http://127.0.0.1:8000/extract/batch', json={
#     "messages":[{"message": """
# (광고)[SKT] 9월 T day 혜택 안내__2025년 9월 17일(수)_럭키찬스!_아이폰 출시 기념 퀴즈_올리브영 기프트 카드 1만 원 응모 이벤트__▶ 자세히 보기: https://t-mms.kr/t.do?m=#61&s=33670&a=&u=https://bit.ly/467rn3q__■ 문의: SKT 고객센터(1558, 무료)__무료 수신거부 1504

# """,
# "message_id": "test_message_id_01"
# },
# {"message": """
# (광고)[SKT] 9월 T day 혜택 안내__2025년 9월 17일(수)_럭키찬스!_아이폰 출시 기념 퀴즈_올리브영 기프트 카드 1만 원 응모 이벤트__▶ 자세히 보기: https://t-mms.kr/t.do?m=#61&s=33670&a=&u=https://bit.ly/467rn3q__■ 문의: SKT 고객센터(1558, 무료)__무료 수신거부 1504

# """,
# "message_id": "test_message_id_02"
# }],
#     "llm_model": "ax",  # CLI와 동일하게 Gemini 사용 (API에서는 'gemini'로 표기)
#     "product_info_extraction_mode": "llm",
#     "entity_matching_mode": "llm",  # 올바른 파라미터 이름
#     "extract_entity_dag": False,
#     "result_type": "ext",
#     "save_to_mongodb": True,
    
# })

result = response.json()

print(json.dumps(result, indent=4, ensure_ascii=False))

# response = requests.post('http://127.0.0.1:8000/dag', json={
#     "message": """광고 제목:[SK텔레콤] 3월 0 day 혜택 안내
# 광고 내용:(광고)[SKT] 2월 0 day 혜택 안내__[2월 10일(토) 혜택]_만 13~34세 고객이라면_베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!_(선착순 3만 명 증정)_▶ 자세히 보기: http://t-mms.kr/t.do?m=#61&s=24589&a=&u=https://bit.ly/3SfBjjc__■ 에이닷 X T 멤버십 시크릿코드 이벤트_에이닷 T 멤버십 쿠폰함에 ‘에이닷이빵쏜닷’을 입력해보세요!_뚜레쥬르 데일리우유식빵 무료 쿠폰을 드립니다._▶ 시크릿코드 입력하러 가기: https://bit.ly/3HCUhLM__■ 문의: SKT 고객센터(1558, 무료)_무료 수신거부 1504""",
#   "llm_model": "ax",
#   "save_dag_image": True,
#   "save_to_mongodb": True,
#   "message_id": "test_message_id_dag"
# })

# result = response.json()

# print(json.dumps(result, indent=4, ensure_ascii=False))

# # DAG 이미지 URL 확인 (외부 시스템에서 접근 가능)
# if result.get('success') and result.get('result', {}).get('dag_image_url'):
#     print("\n" + "="*80)
#     print("📊 DAG 이미지 URL (외부 시스템 접근 가능):")
#     print(result['result']['dag_image_url'])
#     print("="*80)

# print(json.dumps(result['result'], indent=4, ensure_ascii=False))


