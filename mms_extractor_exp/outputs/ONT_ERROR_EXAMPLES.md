# ONT Entity Extraction Error Examples

This document provides concrete examples for each error category to help understand extraction issues.

---

## FALSE POSITIVE EXAMPLES

### 1. Benefit Category (35-36 cases)

These are promotional/benefit descriptions that should NOT be extracted as standalone entities:

**Row 2 - iPhone 17 사전예약 메시지:**
```
❌ AX FP: "T다이렉트샵 특별 사은품"
❌ AX FP: "[사죠영] 죠르디 한정판 기프트"
❌ AX FP: "SKT 고객센터(1558, 무료)"
```
→ These are gift/benefit descriptions, not core products/services

**Row 5 - 새샘대리점 메시지:**
```
❌ AX FP: "최대 보상"
❌ AX FP: "풍성한 사은품"
❌ AX FP: "최대 할인"
❌ AX FP: "월정액 25% 추가 할인"
❌ AX FP: "10월 혜택"
❌ CLD FP: "스마트폰 할인"
```
→ Promotional language, not entities

**Row 6 - 티원대리점 메시지:**
```
❌ AX FP: "T 즉시보상 최대 70% 혜택"
❌ AX FP: "액정 보호 필름 무료 교체"
❌ AX FP: "키친타월 증정"
❌ AX FP: "제휴 카드 추가 할인"
```
→ Store-specific promotions/gifts

**Row 12 - iPhone 신제품 혜택:**
```
❌ AX FP: "최대 22만 원 캐시백"
❌ AX FP: "티머니 충전 쿠폰 96만 원"
```
→ Cash incentive descriptions

---

### 2. Channel Category (8 AX, 24 CLD - CLD 3배!)

These are communication channels, not products:

**Row 1 - 에이닷 알람 메시지:**
```
❌ CLD FP: "에이닷 앱"
```

**Row 2 - iPhone 사전예약:**
```
❌ CLD FP: "카카오톡 상담"
❌ CLD FP: "SKT 고객센터"
```

**Row 4 - T우주 wavve:**
```
❌ CLD FP: "SKT 구독상품 전담 고객센터(1505)"
❌ CLD FP: "wavve 앱"
```

**Row 5 - 새샘대리점:**
```
❌ CLD FP: "SKT 고객센터(1558)"
❌ CLD FP: "매장 홈페이지"
```

**Row 7 - 투썸플레이스:**
```
❌ AX FP: "https://t-mms.kr/aL9/#74"
❌ CLD FP: "https://t-mms.kr/aL9/#74"
❌ CLD FP: "T 우주 고객센터(1505)"
```

**Row 10 - T day 혜택:**
```
❌ CLD FP: "https://bit.ly/467rn3q"
```

**Row 14 - 신분증 진위확인:**
```
❌ AX FP: "T 월드 매장"
❌ CLD FP: "T 월드 매장"
```

→ **CLD extracts far more channels than AX** - needs stricter filtering

---

### 3. Other Category (39-51 cases)

Miscellaneous issues including spacing, partner brands, over-detailed specs:

**Row 1 - 스페이싱 이슈:**
```
❌ AX FP: "A. (에이닷)"  ← space after dot
✅ Correct: "A.(에이닷)"  ← no space
```

**Row 1 - 파트너 브랜드:**
```
❌ AX FP: "CU"
❌ AX FP: "CU 빙그레 바나나우유 기프티콘"
❌ AX FP: "이든앤앨리스"
```

**Row 2 - 선물 브랜드들:**
```
❌ CLD FP: "사죠영"
❌ CLD FP: "크레앙"
❌ CLD FP: "프리디"
❌ CLD FP: "에이프릴스톤"
```

**Row 2 - 이벤트명:**
```
❌ CLD FP: "아이폰17/ 17 Pro 사전예약"
```

**Row 5 - 과도한 상세정보:**
```
❌ AX FP: "갤럭시 Z 플립7(512GB)"  ← includes capacity
❌ AX FP: "기가라이트 인터넷(최대 500Mbps)"  ← includes speed
❌ AX FP: "180개 고화질 TV 채널"  ← feature description
✅ Correct: "갤럭시 Z 플립7"
✅ Correct: "기가라이트 인터넷"
```

**Row 8 - 복합 엔티티:**
```
❌ AX FP: "공신폰/부모님폰"  ← should be split
✅ Correct: "공신폰", "부모님폰"
```

**Row 9 - 채널 타입:**
```
❌ AX FP: "SK텔레콤 공식인증대리점"
❌ AX FP: "갤럭시 폴더블/퀀텀"  ← category, not product
```

**Row 10 - 시간 접두사:**
```
❌ AX FP: "9월 T day"  ← includes month
✅ Correct: "T day"
```

**Row 15 - 시간 접두사:**
```
❌ AX FP: "2월 0 day"  ← includes month
✅ Correct: "0 day"
```

---

### 4. Pricing/Amount Category (2-1 cases)

Extracting price as entity:

**Row 5 - 새샘대리점:**
```
❌ AX FP: "월 이용요금 3만 원대"
```

**Row 7 - 투썸플레이스:**
```
❌ AX FP: "월 1,900원"
```

→ Prices should be attributes, not entities

---

### 5. Segment/Condition Category (2-5 cases)

Extracting eligibility/targeting criteria:

**Row 12 - iPhone 신제품:**
```
❌ AX FP: "애플 액세서리 팩"
```

**Row 13 - 티다문구점:**
```
❌ CLD FP: "신규가입 고객"
```

**Row 14 - 신분증 진위확인:**
```
❌ CLD FP: "신분증 정보 불일치 회선 고객"
```

**Row 15 - 2월 0 day:**
```
❌ AX FP: "만 13~34세"
❌ CLD FP: "만 13~34세"
```

→ Customer segments/conditions, not products

---

### 6. Generic/Vague Category (1 case)

**Row 2 - iPhone 사전예약:**
```
❌ AX FP: "5G"
```

→ Too generic when standalone

---

## FALSE NEGATIVE EXAMPLES

### 1. Entity Partially Captured (17-20 cases)

The entity exists but in a different format:

**Row 2 - 상품명 분리:**
```
✅ Correct: "아이폰 17/17 Pro"
❌ AX Extracted: "아이폰 17", "아이폰 17 Pro" (separated)
```

**Row 3 - 일반명사 누락:**
```
✅ Correct: "부스트 파크"
❌ CLD Extracted: (only in benefit phrase "부스트 파크 특별 혜택")
```

**Row 5 - 상세정보 과다/누락:**
```
✅ Correct: "5GX 프리미엄 요금제"
❌ Extracted: "5GX 프리미엄" (누락)

✅ Correct: "갤럭시 Z 플립7"
❌ Extracted: "갤럭시 Z 플립7(512GB 용량 업그레이드)" (과다)

✅ Correct: "기가라이트 인터넷"
❌ Extracted: "기가라이트 인터넷(최대 500Mbps)" (과다)
```

**Row 6 - 일반화/구체화:**
```
✅ Correct: "아이폰 신제품"
❌ CLD Extracted: "아이폰" (too generic)
```

**Row 7 - 분리 추출:**
```
✅ Correct: "투썸플레이스 20% 할인"
❌ Extracted: "투썸플레이스", "20% 할인" (separated)
```

**Row 8 - 복합 vs 분리:**
```
✅ Correct: "공신폰", "부모님폰" (separate)
❌ Extracted: "공신폰/부모님폰" (combined)
```

**Row 9 - 일반명사 포함/제외:**
```
✅ Correct: "퀀텀"
❌ Extracted: "갤럭시 퀀텀" (too specific)

✅ Correct: "갤럭시 폴더블"
❌ Extracted: "갤럭시 폴더블/퀀텀" (combined)

✅ Correct: "공식인증대리점"
❌ Extracted: "SK텔레콤 공식인증대리점" (too specific)

✅ Correct: "T 월드"
❌ Extracted: "T 월드 매장" (too specific)
```

**Row 10 - 시간 접두사:**
```
✅ Correct: "T day"
❌ Extracted: "9월 T day" (includes month)

✅ Correct: "아이폰"
❌ Extracted: (only in "아이폰 출시 기념 퀴즈")
```

**Row 11 - 상품명 분해:**
```
✅ Correct: "이모션캐슬 쿠폰팩"
❌ Extracted: "T 우주 이모션캐슬 쿠폰팩" (too specific)

✅ Correct: "티니핑 캐릭터 상품"
❌ Extracted: "티니핑" (too generic)
```

**Row 13 - 복합 엔티티:**
```
✅ Correct: "갤럭시 Z 폴드7|Z 플립7 액세서리"
❌ Extracted: "갤럭시 Z 폴드7", "갤럭시 Z 플립7" (separated, missing "액세서리")
```

**Row 14 - 상세정보 과다:**
```
✅ Correct: "신분증 진위확인"
❌ Extracted: "신분증 진위확인 절차" (too detailed)

✅ Correct: "T 월드 매장 방문"
❌ Extracted: "T 월드 매장" (missing action)
```

**Row 15 - 시간 접두사 & 분해:**
```
✅ Correct: "0 day"
❌ Extracted: "2월 0 day" (includes month)

✅ Correct: "시크릿코드"
❌ Extracted: "에이닷 X T 멤버십 시크릿코드 이벤트" (too long)

✅ Correct: "에이닷"
❌ CLD Extracted: "에이닷 앱" (too specific)
```

---

### 2. Completely Missed (1-2 cases)

**Row 1 - 스페이싱 이슈:**
```
✅ Correct: "A.(에이닷)"
❌ Both models extracted: "A. (에이닷)" instead
→ Space after dot caused mismatch
```

**Row 8 - 일반명사 필터링 (CLD only):**
```
✅ Correct: "아이폰 신제품"
❌ CLD: Completely missed (probably filtered as too generic)
```

---

## SUMMARY OF ISSUES

### Format Normalization Problems

1. **Spacing**: "A.(에이닷)" vs "A. (에이닷)"
2. **Detail level**: "5GX 프리미엄" vs "5GX 프리미엄 요금제"
3. **Spec inclusion**: "갤럭시 Z 플립7" vs "갤럭시 Z 플립7(512GB)"
4. **Compound splitting**: "공신폰/부모님폰" vs ["공신폰", "부모님폰"]
5. **Temporal prefixes**: "T day" vs "9월 T day"

### Conceptual Extraction Problems

1. **Benefit language**: Extracting promotional phrases as entities
2. **Channel information**: CLD extracts 3x more than AX
3. **Partner brands**: Gift/accessory brands extracted
4. **Price as entity**: Treating amounts as entities
5. **Customer segments**: Extracting eligibility criteria

### Model Comparison

**AX Characteristics:**
- Fewer false positives (87 vs 117)
- Less aggressive on channels (8 vs 24)
- Similar on benefits (35 vs 36)

**CLD Characteristics:**
- More false positives overall (+34%)
- Much more aggressive on channels (+200%)
- Extracts more detailed/specific forms
- More false negatives (+22%)

---

## RECOMMENDED FIXES

### For False Positives (Priority: High)

1. **Block benefit keywords**: 할인, 증정, 무료, 캐시백, 보상, 혜택, 특가, 이벤트
2. **Filter customer service**: 고객센터, 상담, 문의
3. **Remove URLs and apps**: http, https, 앱
4. **Filter gift brands**: Unless part of main product offering

### For False Negatives (Priority: Medium)

1. **Normalize spacing**: Standardize Korean/English boundaries
2. **Strip specs**: Remove (용량), (속도) details
3. **Remove temporal prefixes**: Strip "N월" from recurring entities
4. **Split compounds**: Separate "/" delimited entities when appropriate
