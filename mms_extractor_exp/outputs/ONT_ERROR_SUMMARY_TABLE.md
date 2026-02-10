# ONT Entity Extraction Error Summary

## False Positive Categories

| Category | AX Count | CLD Count | Examples | Impact |
|----------|----------|-----------|----------|--------|
| **Benefit** | **35** | **36** | "최대 22만 원 캐시백", "무료 대여", "월정액 25% 추가 할인", "키친타월 증정" | **Critical** - Extracting promotional language instead of product names |
| **Channel** | 8 | **24** | "SKT 고객센터(1558)", "매장 홈페이지", "카카오톡 상담", "에이닷 앱" | **High** - CLD extracts 3x more channels than AX |
| **Other** | **39** | **51** | "A. (에이닷)" (spacing), "사죠영/크레앙" (gift brands), "공신폰/부모님폰" | **Medium-High** - Mixed issues: format, partner brands, over-detail |
| **Pricing/Amount** | 2 | 1 | "월 이용요금 3만 원대", "월 1,900원" | **Low** - Treating prices as entities |
| **Segment/Condition** | 2 | 5 | "만 13~34세", "신규가입 고객", "애플 액세서리 팩" | **Medium** - Extracting eligibility criteria |
| **Generic/Vague** | 1 | 0 | "5G" | **Low** - Very generic terms |
| **TOTAL** | **87** | **117** | | CLD has **34% more false positives** |

---

## False Negative Categories

| Category | AX Count | CLD Count | Examples | Impact |
|----------|----------|-----------|----------|--------|
| **Entity partially captured** | **17** | **20** | "아이폰 17/17 Pro" vs "아이폰 17", "5GX 프리미엄 요금제" vs "5GX 프리미엄" | **Medium** - Format normalization issues |
| **Completely missed** | 1 | 2 | "A.(에이닷)" (spacing), "아이폰 신제품" (CLD only) | **Low** - Very few complete misses |
| **TOTAL** | **18** | **22** | | Recall is good, normalization needs work |

---

## Key Statistics

### Overall Performance
- **AX:** 87 false positives, 18 false negatives
- **CLD:** 117 false positives (+34%), 22 false negatives (+22%)

### Category Distribution (False Positives)

| Category | AX % | CLD % |
|----------|------|-------|
| Other | 44.8% | 43.6% |
| Benefit | 40.2% | 30.8% |
| Channel | 9.2% | 20.5% |
| Segment | 2.3% | 4.3% |
| Pricing | 2.3% | 0.9% |
| Generic | 1.1% | 0% |

### Category Distribution (False Negatives)

| Category | AX % | CLD % |
|----------|------|-------|
| Partial capture | 94.4% | 90.9% |
| Completely missed | 5.6% | 9.1% |

---

## Main Problem: Over-Extraction

**Both models extract far too many entities**, especially:

1. **Benefit descriptions** (35-36 cases)
   - Discount amounts
   - Gift descriptions
   - Promotional language
   - Customer service info

2. **Channel/Contact info** (8-24 cases)
   - CLD particularly aggressive (3x more than AX)
   - URLs, phone numbers, app names

3. **Miscellaneous entities** (39-51 cases)
   - Partner brands
   - Event names
   - Over-detailed specs

---

## Recommended Actions

### 🔴 High Priority (Fix 60-70% of false positives)

1. **Filter Benefit Language**
   - Block discount/promotion phrases
   - Remove customer service contacts
   - Filter gift descriptions

2. **Reduce CLD Channel Extraction**
   - Configure stricter rules for CLD
   - Filter URLs and contact info
   - Remove app/website references

3. **Entity Normalization**
   - Standardize spacing
   - Remove spec details (512GB, 500Mbps)
   - Split/merge compound entities

### 🟡 Medium Priority

4. Partner brand filtering policy
5. Event name extraction rules
6. Temporal prefix removal

### 🟢 Low Priority

7. Generic term filtering
8. Pricing entity policy clarification

---

## Conclusion

**The main issue is PRECISION, not RECALL.**

- Models extract too many non-product entities
- CLD is 34% more aggressive than AX
- False negatives are mostly format variations (easy to fix)
- Focus should be on **filtering false positives** rather than improving detection
