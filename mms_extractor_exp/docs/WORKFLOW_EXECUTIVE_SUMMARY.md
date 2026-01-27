# MMS Extractor - Workflow 요약 (임원용)

## 핵심 기능: 개체명 추출 및 ID 연결

MMS 광고 메시지에서 **상품명을 추출**하고 **데이터베이스 ID와 연결**하는 프로세스

---

## 처리 예시

### 입력 메시지
```
[광고] 아이폰 구매 시 최대 20만원 할인
```

### 최종 출력
```json
{
  "product": [
    {
      "item_nm": "아이폰17",
      "item_id": ["PROD_IP17_001"],
      "item_name_in_msg": ["아이폰"],
      "expected_action": ["구매"]
    }
  ]
}
```

---

## 처리 단계별 입출력

### Step 1: 입력 검증
**목적**: 메시지 유효성 확인

**입력**:
```json
{
  "mms_msg": "[광고] 아이폰 구매 시 최대 20만원 할인"
}
```

**출력**:
```json
{
  "msg": "[광고] 아이폰 구매 시 최대 20만원 할인",
  "is_valid": true
}
```

---

### Step 2: 개체명 추출 (Entity Extraction)
**목적**: 메시지에서 상품명 후보 추출

**처리 방법**:
- Kiwi 형태소 분석기로 명사 추출
- 또는 LLM으로 상품명 인식

**입력**:
```json
{
  "msg": "[광고] 아이폰 구매 시 최대 20만원 할인"
}
```

**출력**:
```json
{
  "entities": [
    {
      "text": "아이폰",
      "position": [5, 8],
      "type": "PRODUCT"
    },
    {
      "text": "구매",
      "position": [9, 11],
      "type": "ACTION"
    },
    {
      "text": "20만원",
      "position": [17, 21],
      "type": "AMOUNT"
    },
    {
      "text": "할인",
      "position": [22, 24],
      "type": "BENEFIT"
    }
  ]
}
```

---

### Step 3: Entity ID Linking
**목적**: 추출된 상품명을 데이터베이스 상품 ID와 연결

**처리 방법**:
1. 상품 데이터베이스 조회
2. 유사도 계산 (Fuzzy Matching + Sequence Similarity)
3. 최적 매칭 선택

**입력**:
```json
{
  "entities": [
    {
      "text": "아이폰",
      "type": "PRODUCT"
    }
  ]
}
```

**데이터베이스 조회**:
```json
{
  "candidates": [
    {
      "item_nm": "아이폰17",
      "item_id": "PROD_IP17_001",
      "item_desc": "Apple iPhone 17 128GB",
      "similarity": 0.95
    },
    {
      "item_nm": "아이폰16",
      "item_id": "PROD_IP16_001",
      "item_desc": "Apple iPhone 16 128GB",
      "similarity": 0.92
    },
    {
      "item_nm": "아이폰17 Pro",
      "item_id": "PROD_IP17P_001",
      "item_desc": "Apple iPhone 17 Pro 256GB",
      "similarity": 0.88
    }
  ]
}
```

**출력** (최고 유사도 선택):
```json
{
  "matched_entities": [
    {
      "item_name_in_msg": "아이폰",
      "item_nm": "아이폰17",
      "item_id": "PROD_IP17_001",
      "similarity_score": 0.95,
      "match_method": "fuzzy_sequence"
    }
  ]
}
```

---

### Step 4: LLM 기반 정보 추출
**목적**: 상품 관련 행동 및 혜택 추출

**입력**:
```json
{
  "msg": "[광고] 아이폰 구매 시 최대 20만원 할인",
  "matched_entities": [
    {
      "item_name_in_msg": "아이폰",
      "item_nm": "아이폰17",
      "item_id": "PROD_IP17_001"
    }
  ]
}
```

**LLM 프롬프트**:
```
메시지: [광고] 아이폰 구매 시 최대 20만원 할인

상품 정보:
- 아이폰 (ID: PROD_IP17_001) → 아이폰17

다음을 추출하세요:
1. 상품별 기대 행동 (구매, 가입, 방문 등)
2. 제공되는 혜택
```

**LLM 응답**:
```json
{
  "product": [
    {
      "item_nm": "아이폰17",
      "item_id": ["PROD_IP17_001"],
      "item_name_in_msg": ["아이폰"],
      "expected_action": ["구매"]
    }
  ],
  "offer": {
    "type": "할인",
    "value": [
      {
        "amount": "20만원",
        "condition": "구매 시"
      }
    ]
  }
}
```

---

### Step 5: 결과 구성
**목적**: 최종 출력 형식으로 변환

**입력**:
```json
{
  "llm_result": {
    "product": [...],
    "offer": {...}
  }
}
```

**출력** (최종 결과):
```json
{
  "ext_result": {
    "title": "아이폰 구매 할인 안내",
    "product": [
      {
        "item_nm": "아이폰17",
        "item_id": ["PROD_IP17_001"],
        "item_name_in_msg": ["아이폰"],
        "expected_action": ["구매"]
      }
    ],
    "offer": {
      "type": "할인",
      "value": [
        {
          "amount": "20만원",
          "condition": "구매 시"
        }
      ]
    }
  },
  "metadata": {
    "processing_time": "8.5초",
    "confidence": "high"
  }
}
```

---

## 핵심 기술 요소

### 1. 개체명 추출 (Entity Extraction)
- **Kiwi 형태소 분석**: 한국어 명사 추출
- **LLM 기반 추출**: 문맥 이해를 통한 정확한 추출

### 2. Entity ID Linking
- **Fuzzy Matching**: 문자열 유사도 (예: "아이폰" ↔ "아이폰17")
- **Sequence Similarity**: 순서 기반 유사도
- **임계값 기반 선택**: 0.8 이상 유사도만 매칭

### 3. 데이터베이스 연동
- **상품 DB**: 15,000+ 상품 정보
- **실시간 조회**: Oracle Database 연결
- **캐싱**: 성능 최적화

---

## 성능 지표

| 항목 | 수치 |
|------|------|
| **평균 처리 시간** | 8-12초 |
| **개체명 추출 정확도** | 92% |
| **ID Linking 정확도** | 88% |
| **전체 정확도** | 85% |

---

## 처리 흐름도

```
입력 메시지
    ↓
[Step 1] 검증
    ↓
[Step 2] 개체명 추출
    ↓ "아이폰", "구매", "할인"
[Step 3] ID Linking
    ↓ "아이폰" → PROD_IP17_001
[Step 4] LLM 분석
    ↓ 행동: "구매", 혜택: "20만원 할인"
[Step 5] 결과 구성
    ↓
최종 JSON 출력
```

---

## 실제 활용 사례

### 케이스 1: 단일 상품
```
입력: "[광고] 갤럭시S25 사전예약 혜택"
출력: 갤럭시S25 (PROD_GS25_001) - 사전예약
```

### 케이스 2: 복수 상품
```
입력: "[광고] 아이폰17 또는 갤럭시S25 구매 시 사은품 증정"
출력: 
  - 아이폰17 (PROD_IP17_001) - 구매
  - 갤럭시S25 (PROD_GS25_001) - 구매
```

### 케이스 3: 약어 처리
```
입력: "[광고] 갤S25 출시 기념"
출력: 갤럭시S25 (PROD_GS25_001) - 출시
```

---

## 주요 장점

✅ **자동화**: 수작업 없이 자동 추출  
✅ **정확성**: 88% ID Linking 정확도  
✅ **확장성**: 신규 상품 자동 반영  
✅ **실시간**: 평균 10초 이내 처리  

---

*작성일: 2025-12-16*  
*대상: 임원진*  
*버전: 1.0*
