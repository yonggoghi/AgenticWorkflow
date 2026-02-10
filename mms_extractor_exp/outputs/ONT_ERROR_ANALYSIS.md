# ONT Entity Extraction Error Analysis

**Analysis Date:** 2026-02-06
**Compared Files:**
- ONT Results: `entity_extraction_eval_ont_20260206_104832.csv`
- Correct Answers: `entity_extraction_eval_reg_baseline.csv`

## Executive Summary

This analysis compares ONT entity extraction results (both ax and cld models) against correct baseline answers across 15 test messages. We categorized all false positives and false negatives to understand systematic extraction errors.

### Overall Statistics

| Metric | AX | CLD |
|--------|----|----|
| **Total False Positives** | 87 | 117 |
| **Total False Negatives** | 18 | 22 |
| **Precision Issues** | High | Higher |
| **Recall Issues** | Low | Moderate |

**Key Finding:** Both models suffer from **over-extraction** (too many false positives) rather than under-extraction. CLD model extracts 34% more false positives than AX.

---

## False Positive Analysis

### 1. Benefit Category (가장 많은 오류)

**AX Count:** 35
**CLD Count:** 36

**Description:** Extracting promotional benefits, discounts, gifts, and other incentive-related text that should not be standalone entities.

**Examples:**
- "T다이렉트샵 특별 사은품" (Row 2)
- "최대 22만 원 캐시백" (Row 12)
- "무료 대여" (Row 3)
- "월정액 25% 추가 할인" (Row 5)
- "키친타월 증정" (Row 6)
- "액정 보호 필름 무료 교체" (Row 6)
- "SKT 고객센터(1558, 무료)" (Row 2, 4, 6)
- "통신 요금 최대 55% 할인" (Row 9)

**Root Cause:** Models are extracting benefit descriptions as entities rather than focusing on core product/service names.

**Impact:** **Critical** - These are descriptive phrases about offers, not actual products/services.

---

### 2. Channel Category (CLD에서 특히 심각)

**AX Count:** 8
**CLD Count:** 24 (3x more than AX!)

**Description:** Extracting communication channels, customer service contacts, URLs, and app names.

**Examples:**
- "SKT 고객센터(1558)" (multiple rows)
- "매장 홈페이지" (Row 5, 8)
- "카카오톡 상담" (Row 2)
- "https://t-mms.kr/aL9/#74" (Row 7)
- "대리점 공식 홈페이지" (Row 6)
- "에이닷 앱" (Row 1, 15)
- "T 우주 고객센터(1505)" (Row 4, 7, 11)
- "wavve 앱" (Row 4)

**Root Cause:** Models treating customer touchpoints as products/services.

**Impact:** **High** - CLD model is particularly aggressive in extracting channel/contact information.

---

### 3. Other Category (가장 큰 카테고리)

**AX Count:** 39
**CLD Count:** 51

**Description:** Miscellaneous extractions including product components, brand names, event names, and other non-core entities.

**Examples:**
- "A. (에이닷)" vs "A.(에이닷)" (spacing issue, Row 1)
- "CU" (partner brand, Row 1)
- "사죠영", "크레앙", "프리디", "에이프림스톤" (gift brands, Row 2)
- "180개 고화질 TV 채널" (service feature, Row 5)
- "갤럭시 Z 플립7(512GB)" (detailed spec, Row 5)
- "공신폰/부모님폰" (customer segment phone, Row 8)
- "SK텔레콤 공식인증대리점" (channel type, Row 9)
- "9월 T day" (temporal event, Row 10)

**Root Cause:** Mixed issues - format variations, over-detailed extractions, partner brands.

**Impact:** **Medium-High** - Many are related entities but not core products.

---

### 4. Pricing/Amount Category

**AX Count:** 2
**CLD Count:** 1

**Description:** Extracting specific pricing information as entities.

**Examples:**
- "월 이용요금 3만 원대" (Row 5)
- "월 1,900원" (Row 7)

**Root Cause:** Treating prices as entities rather than attributes.

**Impact:** **Low** - Small number but conceptually wrong.

---

### 5. Segment/Condition Category

**AX Count:** 2
**CLD Count:** 5

**Description:** Extracting customer segments, eligibility conditions, or event constraints.

**Examples:**
- "만 13~34세" (age requirement, Row 15)
- "애플 액세서리 팩" (bundle condition, Row 12)
- "신규가입 고객" (customer segment, Row 13)
- "신분증 정보 불일치 회선 고객" (specific customer type, Row 14)

**Root Cause:** Extracting targeting/eligibility criteria as entities.

**Impact:** **Medium** - CLD extracts more segmentation info.

---

### 6. Generic/Vague Category

**AX Count:** 1
**CLD Count:** 0

**Description:** Very generic technical terms.

**Examples:**
- "5G" (Row 2)

**Root Cause:** Extracting overly generic technology terms.

**Impact:** **Low** - Very few cases.

---

## False Negative Analysis

### 1. Entity Partially Captured (Variant Form)

**AX Count:** 17
**CLD Count:** 20

**Description:** The correct entity exists but in a different form (spacing, punctuation, detail level).

**Examples:**
- Correct: "아이폰 17/17 Pro" vs Extracted: "아이폰 17", "아이폰 17 Pro" (Row 2)
- Correct: "5GX 프리미엄 요금제" vs Extracted: "5GX 프리미엄" (Row 5)
- Correct: "갤럭시 Z 플립7" vs Extracted: "갤럭시 Z 플립7(512GB 용량 업그레이드)" (Row 5)
- Correct: "기가라이트 인터넷" vs Extracted: "기가라이트 인터넷(최대 500Mbps)" (Row 5)
- Correct: "부모님폰", "공신폰" vs Extracted: "공신폰/부모님폰" (Row 8)
- Correct: "투썸플레이스 20% 할인" vs Extracted: "투썸플레이스", "20% 할인" (Row 7)
- Correct: "T day" vs Extracted: "9월 T day" (Row 10)
- Correct: "0 day" vs Extracted: "2월 0 day" (Row 15)

**Root Cause:** Entity normalization issues - models are either over-specifying or under-specifying entities.

**Impact:** **Medium** - These are near-misses rather than complete failures.

---

### 2. Completely Missed

**AX Count:** 1
**CLD Count:** 2

**Description:** Correct entities that were not extracted at all.

**Examples:**
- "A.(에이닷)" (Row 1) - Both models extracted variant "A. (에이닷)" instead
- "아이폰 신제품" (Row 8, CLD only) - CLD failed to extract this generic term

**Root Cause:**
- Row 1: Spacing variation ("A." vs "A. ")
- Row 8: Generic term filtering by CLD

**Impact:** **Low** - Very few complete misses.

---

## Key Findings

### 1. Over-Extraction is the Primary Issue

Both models extract far too many entities:
- **Benefit descriptions** (35-36 cases) - Should focus on product names, not promotional language
- **Channel/Contact info** (8-24 cases) - CLD particularly aggressive here
- **Miscellaneous** (39-51 cases) - Partner brands, event names, service features

### 2. CLD Model More Aggressive Than AX

| Category | AX | CLD | Difference |
|----------|----|----|-----------|
| Total FP | 87 | 117 | +30 (+34%) |
| Channel | 8 | 24 | +16 (+200%) |
| Other | 39 | 51 | +12 (+31%) |
| Segment | 2 | 5 | +3 (+150%) |

**CLD extracts significantly more false positives**, especially for channels and customer service information.

### 3. Format Normalization Issues

The models struggle with:
- Spacing variations: "A.(에이닷)" vs "A. (에이닷)"
- Specificity levels: "5GX 프리미엄" vs "5GX 프리미엄 요금제"
- Grouping: "공신폰/부모님폰" vs separate entities
- Detail inclusion: "갤럭시 Z 플립7" vs "갤럭시 Z 플립7(512GB)"

### 4. False Negatives Are Mostly Near-Misses

- Only 1-2 completely missed entities
- 17-20 entities extracted in variant forms
- Recall is actually good, but normalization needs improvement

---

## Recommendations

### High Priority

1. **Filter Benefit Language** - Add rules to prevent extraction of:
   - Discount percentages and amounts
   - Gift/free offer descriptions
   - Customer service contact info
   - Promotional event names

2. **Reduce CLD Channel Extraction** - Configure CLD to be less aggressive with:
   - URLs
   - Customer service numbers
   - App names that are just channels
   - Website references

3. **Entity Normalization** - Implement post-processing to:
   - Standardize spacing (especially for Korean/English mixed text)
   - Remove specification details like capacity, speed
   - Merge compound entities (e.g., split "공신폰/부모님폰")

### Medium Priority

4. **Partner Brand Filtering** - Decide whether gift brands (사죠영, 크레앙, etc.) should be extracted

5. **Event Name Filtering** - Decide policy on extracting event names like "A. 알람 챌린지"

6. **Temporal Prefix Removal** - Strip month/date prefixes from recurring entities (e.g., "9월 T day" → "T day")

### Low Priority

7. **Generic Term Filtering** - Consider filtering very generic terms like "5G" when standalone

8. **Pricing Entity Policy** - Clarify whether pricing info should ever be extracted as entities

---

## Conclusion

**Precision is the main problem, not recall.** Both ONT models are extracting too many entities, with CLD being more aggressive than AX. The primary fix should be **filtering out benefit language and channel information** rather than improving entity detection.

The false negatives are mostly formatting issues that can be addressed through post-processing normalization rather than model changes.

**Recommended Focus:**
1. Reduce false positives by 60-70% through filtering
2. Improve entity normalization for the remaining 15-20% format mismatches
3. CLD model needs stricter extraction rules than AX
