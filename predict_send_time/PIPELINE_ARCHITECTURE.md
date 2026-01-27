# MMS Click Prediction Pipeline - Architecture Reference

> **목적**: 이 문서는 AI 어시스턴트가 코드를 이해하고 수정할 수 있도록 파이프라인 구조를 설명합니다.
> 
> **대상 파일**: `predict_ost_zpln.scala` (Zeppelin Notebook 형식의 Scala 코드)

---

## 🚨 AI Assistant를 위한 빠른 시작 가이드

### 이 문서를 읽는 방법

**처음 접근 시** (코드 전체 이해):
1. **Section 1**: 전체 파이프라인 개요 → 5대 작업 흐름 파악
2. **Section 2**: 핵심 설계 원칙 → 시간 조건 변수 중앙 관리 이해
3. **Section 3**: Paragraph 구조 이해 → 각 그룹의 역할
4. **Section 4**: 데이터 흐름 → DataFrame 추적 및 저장/로딩 패턴

**특정 작업 수행 시** (빠른 참조):
- **시간 범위 변경**: Section 2.1 → Section 6.1
- **Feature 추가**: Section 5.2 → Section 3.2 (Group D)
- **버전 관리**: Section 2.2 → Section 5.1
- **메모리 문제**: Section 2.3 → Section 7.2
- **트러블슈팅**: Section 7

### 핵심 원칙 (반드시 기억)

1. ⭐ **시간 조건 변수는 Paragraph 3에서만 수정**
2. ⭐ **버전 일관성 확인 필수** (저장 버전 = 로딩 버전)
3. ⭐ **데이터 흐름 순서 준수** (Raw → Transformed → Model → Prediction)
4. ⭐ **Suffix 배치 처리로 메모리 절약**

### 작업별 Quick Jump

| 작업 | 참조 섹션 | 관련 Paragraph |
|------|----------|---------------|
| 시간 범위 변경 | 2.1, 6.1 | P3 |
| Feature 추가 | 3.2 (Group D), 5.2 | P10-P13 |
| Pipeline 수정 | 3.2 (Group F) | P17-P19 |
| 모델 튜닝 | 3.2 (Group G) | P22-P26 |
| 버전 업그레이드 | 5.1, 6.2 | P3 |
| 성능 최적화 | 2.3, 7.3 | 전체 |

---

## 1. 전체 파이프라인 개요

### 1.1 파이프라인 목적
- MMS 캠페인 메시지에 대한 사용자 클릭 예측
- 최적 발송 시간대(9시~18시) 결정을 위한 propensity score 계산
- 대용량 데이터 처리 (Spark 기반)

### 1.2 5대 작업 흐름 (Workflow)

```
작업 흐름 1-2: Raw Data 생성
├─ Paragraph 3: 시간 조건 변수 설정
├─ Paragraph 4: Response data 로딩
├─ Paragraph 5: Response data 필터링
├─ Paragraph 6: User feature 로딩 (MMKT)
├─ Paragraph 7: Train/Test split
├─ Paragraph 8-9: Undersampling
├─ Paragraph 10: App usage data 로딩
├─ Paragraph 11: Historical click count
├─ Paragraph 12-13: Feature join & 변환
└─ Paragraph 14: Raw data 저장

작업 흐름 3: Transformed Data 생성
├─ Paragraph 15: Raw data 로딩
├─ Paragraph 16: Prediction dataset 준비
├─ Paragraph 17: Pipeline 파라미터 설정
├─ Paragraph 18: Pipeline 함수 정의
├─ Paragraph 19: Pipeline fitting & transformation
└─ Paragraph 20: Transformer & transformed data 저장

작업 흐름 4: 모델 학습 및 평가
├─ Paragraph 21: Transformed data 로딩
├─ Paragraph 22-23: Model 정의
├─ Paragraph 24-26: 모델 학습 (Click, Gap, Utility)
├─ Paragraph 27: 예측 수행
└─ Paragraph 28-30: 평가

작업 흐름 5: 실제 서비스 예측
├─ Paragraph 31: Propensity score 계산
└─ Paragraph 32: Propensity score 로딩
```

---

## 2. 핵심 설계 원칙

### 2.1 시간 조건 변수 중앙 관리
**위치**: Paragraph 3 (Date Range and Period Configuration)

**핵심 개념**: 모든 작업 흐름의 시간 조건을 한 곳에서 관리하여 일관성 보장

**주요 변수 카테고리**:

```scala
// 작업 흐름 1-2: Raw Data 생성
val sendMonth = "YYYYMM"              // 기준 월
val featureMonth = "YYYYMM"           // 피처 추출 월
val period = N                        // 기간 (개월)
val predictionDTSta = "YYYYMMDD"      // Test 시작일
val predictionDTEnd = "YYYYMMDD"      // Test 종료일
val startHour = 9                     // 시간대 시작
val endHour = 18                      // 시간대 종료

// 작업 흐름 3: Transformed Data
val transformRawDataVersion = "N"     // Raw data 버전
val transformedTrainSaveVersion = "N" // 저장 버전
val transformSuffixGroupSize = N      // 배치 크기

// 작업 흐름 4: 모델 학습
val modelTrainDataVersion = "N"       // 로딩 버전
val modelTestDataVersion = "N"        // 로딩 버전

// 작업 흐름 5: 서비스 예측
val predDT = "YYYYMMDD"               // 예측 날짜
val predSuffixGroupSize = N           // 배치 크기
val predOutputPath = "path"           // 저장 경로
```

### 2.2 데이터 저장/로딩 패턴

#### 패턴 1: 버전 관리된 저장
```scala
// 저장 (Paragraph N)
.parquet(s"aos/sto/dataName${version}")

// 로딩 (Paragraph N+1)
spark.read.parquet(s"aos/sto/dataName${version}")
```

#### 패턴 2: Suffix 기반 배치 처리
```scala
// 메모리 효율을 위해 suffix별로 분할 처리
(0 to 15).map(_.toHexString).grouped(groupSize).foreach { suffixGroup =>
  // suffixGroup: Array("0", "1", "2", ...) 처리
}
```

#### 패턴 3: 동적 파티션 덮어쓰기
```scala
.write
.mode("overwrite")
.partitionBy("send_ym", "send_hournum_cd", "suffix")
.parquet(path)
```

### 2.3 메모리 최적화 전략

1. **캐싱 레벨**: `StorageLevel.MEMORY_AND_DISK_SER` (직렬화하여 메모리 절약)
2. **명시적 unpersist**: 사용 완료된 DataFrame은 즉시 해제
3. **Repartition**: 조인 전 적절한 파티션 수로 재분배
4. **Checkpoint**: 매우 큰 데이터는 checkpoint 사용

---

## 3. Paragraph 구조 이해

### 3.1 Paragraph 명명 규칙

```
// ===== Paragraph N: [제목] (ID: paragraph_XXXXX) =====
```

- **N**: Paragraph 번호 (실행 순서와 일치하지 않을 수 있음)
- **제목**: 기능 설명
- **ID**: Zeppelin notebook 고유 ID

### 3.2 주요 Paragraph 그룹

#### Group A: 설정 및 유틸리티 (P1-P2)
- **P1**: Import 및 Spark 설정
- **P2**: Helper 함수 정의 (`getPreviousMonths`, `getDaysBetween` 등)

#### Group B: 시간 조건 관리 (P3)
- **핵심**: 모든 시간 변수를 여기서 정의
- **검증**: 버전 불일치 자동 검사

#### Group C: 데이터 로딩 및 전처리 (P4-P7)
- **P4**: Response data 로딩
- **P5**: Response data 필터링
- **P6**: User feature (MMKT) 로딩
- **P7**: Train/Test split

#### Group D: Feature Engineering (P8-P13)
- **P8-P9**: Class imbalance 해소 (undersampling)
- **P10**: App usage data 로딩 (대용량)
- **P11**: Historical click count 계산
- **P12**: Multi-way join
- **P13**: 데이터 타입 변환

#### Group E: Raw Data 저장/로딩 (P14-P16)
- **P14**: Train/Test raw data 저장
- **P15**: Raw data 로딩 (transformation용)
- **P16**: Prediction dataset 준비

#### Group F: Feature Transformation (P17-P20)
- **P17**: Pipeline 파라미터
- **P18**: Pipeline 함수 (`makePipeline`)
- **P19**: Pipeline fitting & transformation
- **P20**: Transformer & transformed data 저장

#### Group G: 모델 학습 (P21-P26)
- **P21**: Transformed data 로딩
- **P22-P23**: Model 정의 (GBT, XGBoost, LightGBM 등)
- **P24**: Click 모델 학습
- **P25**: Gap 모델 학습
- **P26**: Utility 모델 학습

#### Group H: 평가 및 예측 (P27-P32)
- **P27**: Test set 예측
- **P28-P30**: 성능 평가
- **P31**: Propensity score 계산 (서비스용)
- **P32**: Propensity score 검증

---

## 4. 데이터 흐름 (Data Flow)

### 4.1 주요 DataFrame 추적

```
resDF (P4: Response data)
  └─> resDFFiltered (P5: 필터링)
       └─> resDFSelected (P7: 피처 추가)
            ├─> resDFSelectedTr (P7: Train)
            │    └─> resDFSelectedTrBal (P9: Undersampled)
            └─> resDFSelectedTs (P7: Test)

mmktDF (P6: User features)
  └─> mmktDFFiltered (P12: 조인용)

xdrDF (P10: App usage, hourly)
  └─> xdrDFMon (P10: Pivot by hour)
  └─> xdrAggregatedFeatures (P10: Summary features)

trainDF (P12: Feature join)
  └─> trainDFRev (P13: 타입 변환)
       └─> [P14 저장]
            └─> [P15 로딩]
                 └─> transformedTrainDF (P19: Pipeline 적용)
                      └─> [P20 저장]
                           └─> [P21 로딩]
                                └─> [P24-P26 학습]

testDF (P12: Feature join)
  └─> testDFRev (P13: 타입 변환)
       └─> [P14 저장]
            └─> [P15 로딩]
                 └─> transformedTestDF (P19: Pipeline 적용)
                      └─> [P20 저장]
                           └─> [P21 로딩]
                                └─> [P27 평가]

predDF (P16: Prediction data)
  └─> predDFRev (P16: 타입 변환)
       └─> [P31 예측]
            └─> propensityScoreDF (P31: 저장)
                 └─> [P32 검증]
```

### 4.2 저장 경로 패턴

| 데이터 | 저장 Paragraph | 경로 패턴 | 로딩 Paragraph |
|--------|---------------|-----------|---------------|
| Raw Train | P14 | `aos/sto/trainDFRev${version}` | P15 |
| Raw Test | P14 | `aos/sto/testDFRev` | P15 |
| Transformer (Click) | P20 | `aos/sto/transformPipelineXDRClick${version}` | P21 |
| Transformer (Gap) | P20 | `aos/sto/transformPipelineXDRGap${version}` | P21 |
| Transformed Train | P20 | `aos/sto/transformedTrainDFXDR${version}` | P21 |
| Transformed Test | P20 | `aos/sto/transformedTestDFXDF${version}` | P21 |
| Propensity Score | P31 | `aos/sto/propensityScoreDF` | P32 |

---

## 5. 작업 시 주의사항

### 5.1 시간 조건 변수 수정 시

1. **Paragraph 3에서만 수정**: 모든 시간 변수는 P3에 중앙 집중
2. **버전 일관성 확인**: P3 하단의 검증 메시지 확인
3. **의존성 체크**: 
   - `transformRawDataVersion` ↔ P14 저장 버전
   - `transformedTrainSaveVersion` ↔ `modelTrainDataVersion`

### 5.2 새로운 Feature 추가 시

1. **Raw feature 단계** (P10-P13):
   - 새 데이터 소스 로딩 → P10 패턴 참조
   - Feature join → P12에 추가
   - 컬럼 타입 지정 → P13에서 처리

2. **Pipeline 단계** (P17-P19):
   - 컬럼 분류 → P17에서 `tokenCols`, `continuousCols`, `categoryCols` 정의
   - Pipeline 변환 → P18의 `makePipeline` 함수 확인
   - 새 feature가 자동으로 처리되는지 검증

3. **저장 경로 업데이트**:
   - P14, P20에서 버전 변경 고려

### 5.3 대용량 데이터 처리 시

1. **Suffix 배치 크기 조정**:
   ```scala
   val transformSuffixGroupSize = 2  // 메모리 부족 시 1로 감소
   val predSuffixGroupSize = 4       // 메모리 부족 시 2로 감소
   ```

2. **Repartition 수 조정**:
   - 조인 전: `.repartition(200, joinKey)`
   - 저장 전: `.repartition(50)` (small files 방지)

3. **캐시 관리**:
   - 사용 완료된 DataFrame은 `.unpersist()` 호출
   - P15, P31에서 이전 캐시 정리 패턴 참조

### 5.4 코드 읽기 우선순위

**처음 접근 시**:
1. P3: 시간 조건 변수 (전체 흐름 이해)
2. P4-P7: 데이터 로딩 및 split 로직
3. P12: Feature join (어떤 feature들이 있는지)
4. P18: Pipeline 함수 (feature transformation 로직)
5. P24-P26: 모델 학습 코드

**특정 작업 수행 시**:
- Feature 추가: P10 → P12 → P13 → P17 → P18
- 시간 범위 변경: P3만 수정
- 모델 튜닝: P22-P23 (모델 정의) → P24-P26 (하이퍼파라미터)
- 예측 실행: P16 → P31 → P32

---

## 6. 일반적인 작업 패턴

### 6.1 시간 범위 변경

```scala
// P3에서 수정
val sendMonth = "202601"        // 새로운 기준 월
val predictionDTSta = "20251201"  // 새로운 split 날짜
```

### 6.2 버전 업그레이드

```scala
// P3에서 버전 통합 관리
val transformRawDataVersion = "11"
val transformedTrainSaveVersion = "11"
val modelTrainDataVersion = "11"
```

### 6.3 실험적 변경 (안전한 방법)

1. **버전 분리**: 새 버전 번호 사용
2. **Suffix 제한**: 일부 suffix만 처리
   ```scala
   val prdSuffix = "0,1,2"  // 전체 대신 일부만
   ```
3. **샘플링**: P19에서 샘플 비율 조정
   ```scala
   val transformSampleRate = 0.1  // 10%만 사용
   ```

---

## 7. 트러블슈팅 체크리스트

### 7.1 데이터가 없을 때
- [ ] P3의 시간 범위가 실제 데이터와 일치하는가?
- [ ] 저장 경로와 로딩 경로가 일치하는가?
- [ ] 버전 번호가 일치하는가?

### 7.2 메모리 부족 (OOM)
- [ ] Suffix 배치 크기를 줄였는가?
- [ ] 사용하지 않는 DataFrame을 unpersist 했는가?
- [ ] Repartition 수가 적절한가?

### 7.3 성능 저하
- [ ] Broadcast join이 적용되었는가? (작은 테이블)
- [ ] 조인 전 repartition으로 shuffle 최적화했는가?
- [ ] 캐싱이 적절한 StorageLevel로 설정되었는가?

---

## 8. 확장 가능성

### 8.1 새로운 작업 흐름 추가 시

1. **P3에 시간 변수 추가**:
   ```scala
   // 작업 흐름 6: [새 작업]
   val newWorkflowDate = "YYYYMMDD"
   val newWorkflowVersion = "1"
   ```

2. **새 Paragraph 생성**:
   ```scala
   // ===== Paragraph N: [제목] =====
   // =============================================================================
   // [작업 흐름 6] 설명
   // =============================================================================
   // 시간 조건 변수: P3의 newWorkflowDate 사용
   // =============================================================================
   ```

3. **검증 로직 추가**: P3 하단에 버전 체크 추가

### 8.2 모델 추가 시

- P22-P23: 새 모델 정의
- P24-P26: 학습 로직 추가 (기존 패턴 참조)
- P27: 예측 수행
- P28-P30: 평가 메트릭 추가

---

## 9. 코드 스타일 가이드

### 9.1 주석 규칙

```scala
// =============================================================================
// [작업 흐름 N] 간단한 제목
// =============================================================================
// 상세 설명
// - 사용하는 변수
// - 주의사항
// =============================================================================
```

### 9.2 로깅 패턴

```scala
println("=" * 80)
println("[작업 흐름 N] 단계 제목")
println("=" * 80)
println(s"상세 정보: $variable")
println("=" * 80)
```

### 9.3 변수 명명

- 시간 관련: `sendMonth`, `predictionDTSta`, `featureYmList`
- 버전 관련: `transformRawDataVersion`, `modelTrainDataVersion`
- DataFrame: `resDF`, `trainDF`, `transformedTrainDF`
- 경로: `trainSavePath`, `predOutputPath`

---

## 10. Quick Reference

### 자주 사용하는 명령어

```scala
// 시간 범위 생성
getPreviousMonths(startMonth, period)
getDaysBetween(startDay, endDay)

// DataFrame 캐싱
.persist(StorageLevel.MEMORY_AND_DISK_SER)
.checkpoint()

// Suffix 필터링
.filter(s"svc_mgmt_num like '%${suffix}'")

// 배치 처리
(0 to 15).map(_.toHexString).grouped(groupSize)
```

### 중요 경로

- Response data: `aos/sto/response`
- User feature: `wind_tmt.mmkt_svc_bas_f`
- App usage: `dprobe.mst_app_svc_app_monthly`
- Checkpoint: `hdfs://scluster/user/g1110566/checkpoint`

---

## 마지막 업데이트
- **날짜**: 2026-01-22
- **버전**: 1.0
- **작성자**: AI Assistant
- **다음 검토**: 코드 구조 변경 시
