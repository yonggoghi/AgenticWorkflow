# 서비스 데이터 생성 가이드

> 예시: 2026년 3월 3일 서비스 데이터 생성

---

## Case A: 기존 모델 재사용 (Transformer + Model이 이미 있는 경우)

`service_prediction_v3.scala` **Paragraph 3만 수정**하면 됩니다.

```scala
// === 변경할 값 ===
val predDT = "20260303"           // ← 서비스 날짜
val predSendYM = predDT.take(6)   // → "202603" (자동)
val predFeatureYM = getPreviousMonths(predSendYM, 2)(0)  // → "202601" (자동)

// === 기존 모델과 버전 일치 확인 ===
val transformedDataVersion = "1"   // data_transformation에서 사용한 버전
val modelVersion = "1"             // model_training에서 사용한 버전
val trainSendMonth = "202511"      // model_training에서 사용한 학습 기준 월
val trainPeriod = 3
```

경로는 자동 조합됩니다:

| 항목 | 경로 |
|------|------|
| Click Transformer | `aos/sto/transformPipelineClick_v1_202509-202511` |
| Click Model | `aos/sto/pipelineModelClick_gbtc_click_v1_202509-202511` |
| Output | `aos/sto/mms_score` (send_ym=202603) |

---

## Case B: 최신 데이터로 신규 학습 (Step 1 → 2 → 3 순서 실행)

### Step 1: `data_transformation.scala` Paragraph 3

```scala
val rawDataVersion = "1"           // raw_data_generation 버전
val predictionDTSta = "20260201"   // 테스트 기준일 (2월 데이터로 검증)
val predictionDTEnd = "20260301"
// → trainSendMonth  = "202601" (자동)
// → trainSendYmList = ["202511", "202512", "202601"] (3개월, 자동)
// → testSendYmList  = ["202602"] (자동)

val trainPeriod = 3
val transformedDataVersion = "2"   // 새 버전으로 구분
```

출력 경로:

```
aos/sto/transformPipelineClick_v2_202511-202601
aos/sto/transformedTrainDF_v2_202511-202601
aos/sto/transformedTestDF_v2_20260201-20260301
```

---

### Step 2: `model_training.scala` Paragraph 2

```scala
val transformedDataVersion = "2"   // ← Step 1과 동일
val predictionDTSta = "20260201"   // ← Step 1과 동일
val predictionDTEnd = "20260301"   // ← Step 1과 동일
val trainPeriod = 3
val trainSendMonth = "202601"      // ← Step 1 결과와 동일
val modelVersion = "2"             // 새 버전
```

---

### Step 3: `service_prediction_v3.scala` Paragraph 3

```scala
val predDT = "20260303"
// → predSendYM    = "202603"
// → predFeatureYM = "202601"  ← MMKT/XDR Feature 기준월

val transformedDataVersion = "2"   // ← Step 1, 2와 동일
val modelVersion = "2"             // ← Step 2와 동일
val trainSendMonth = "202601"      // ← Step 2와 동일
val trainPeriod = 3
```

---

## 버전 일치 체크리스트

| 변수 | data_transformation | model_training | service_prediction |
|------|--------------------|-----------------|--------------------|
| `transformedDataVersion` | 설정 | 동일하게 맞춤 | 동일하게 맞춤 |
| `modelVersion` | — | 설정 | 동일하게 맞춤 |
| `trainSendMonth` | 자동 계산 | 결과값 확인 후 입력 | 동일하게 맞춤 |
| `trainPeriod` | 설정 | 동일하게 맞춤 | 동일하게 맞춤 |
| `predictionDTSta/End` | 설정 | 동일하게 맞춤 | — |
