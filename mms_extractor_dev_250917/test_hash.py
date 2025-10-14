#!/usr/bin/env python3
"""
DAG 이미지 해시 테스트 스크립트
==============================

샘플 메시지들의 해시값을 계산하고 해당 DAG 이미지 파일이 존재하는지 확인합니다.
"""

import os
import hashlib
from pathlib import Path

def sha256_hash(text):
    """utils.py와 동일한 해시 함수"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# 샘플 메시지들 (demo.html과 동일)
sample_messages = [
    """[SK텔레콤] ZEM폰 포켓몬에디션3 안내
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

수신거부 080-011-0000""",

    """[T world] 5G 요금제 특가 혜택
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

수신거부 080-011-0000""",

    """[SK텔레콤] 갤럭시 S24 사전예약
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
]

def main():
    print("=" * 80)
    print("🔍 DAG 이미지 해시 테스트")
    print("=" * 80)
    
    dag_images_dir = Path('./dag_images')
    
    if not dag_images_dir.exists():
        print("❌ dag_images 디렉토리가 존재하지 않습니다.")
        return
    
    # 기존 DAG 파일들 나열
    existing_files = list(dag_images_dir.glob('dag_*.png'))
    print(f"📁 기존 DAG 이미지 파일 수: {len(existing_files)}")
    
    if existing_files:
        print("\n📋 기존 DAG 이미지 파일들:")
        for i, file_path in enumerate(sorted(existing_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]):
            size_kb = file_path.stat().st_size / 1024
            mtime = file_path.stat().st_mtime
            print(f"  {i+1}. {file_path.name} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 80)
    print("🧪 샘플 메시지 해시 테스트")
    print("=" * 80)
    
    for i, message in enumerate(sample_messages, 1):
        print(f"\n📝 샘플 메시지 {i}:")
        print(f"   제목: {message.split(chr(10))[0]}")
        
        # 해시 계산
        hash_value = sha256_hash(message)
        expected_filename = f"dag_{hash_value}.png"
        expected_path = dag_images_dir / expected_filename
        
        print(f"   해시: {hash_value}")
        print(f"   예상 파일명: {expected_filename}")
        
        # 파일 존재 여부 확인
        if expected_path.exists():
            size_kb = expected_path.stat().st_size / 1024
            print(f"   ✅ 이미지 존재: {size_kb:.1f} KB")
        else:
            print(f"   ❌ 이미지 없음")
    
    print("\n" + "=" * 80)
    print("💡 테스트 완료")
    print("=" * 80)
    print("📌 데모 웹사이트에서 샘플 메시지를 클릭하고 DAG 추출을 활성화하여")
    print("   해당 메시지의 DAG 이미지가 올바르게 표시되는지 확인하세요.")

if __name__ == "__main__":
    main()
