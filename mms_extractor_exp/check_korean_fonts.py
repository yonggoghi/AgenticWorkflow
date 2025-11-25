#!/usr/bin/env python3
"""한글 폰트 확인 스크립트"""
import matplotlib.font_manager as fm

print("=" * 80)
print("시스템에 설치된 한글 폰트 목록")
print("=" * 80)

fonts = [f.name for f in fm.fontManager.ttflist]

# 우선순위별 한글 폰트 목록
preferred_fonts = [
    'AppleGothic',
    'Apple SD Gothic Neo',
    'AppleMyungjo',
    'NanumGothic',
    'NanumMyeongjo',
    'Malgun Gothic',
    'Dotum',
    'Gulim'
]

print("\n[우선순위 폰트 확인]")
for font in preferred_fonts:
    status = "✓ 설치됨" if font in fonts else "✗ 없음"
    print(f"  {status:12} {font}")

# 한글 관련 폰트 검색
korean_fonts = [f for f in fonts if any(k in f for k in ['Apple', 'Nanum', 'Malgun', 'Dotum', 'Gulim', 'Gothic', 'Myeongjo'])]

print(f"\n[검색된 한글 관련 폰트 ({len(set(korean_fonts))}개)]")
for font in sorted(set(korean_fonts))[:20]:
    print(f"  - {font}")

print("\n" + "=" * 80)
