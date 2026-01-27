# MMS Click Prediction Pipeline

> Spark 기반 대규모 MMS 캠페인 클릭 예측 및 최적 발송 시간 추천 시스템

## 📁 프로젝트 구조

```
predict_send_time/
├── predict_ost_zpln.scala          # 메인 파이프라인 코드 (Zeppelin Notebook)
│
├── 📚 Documentation/
│   ├── README.md                   # 프로젝트 개요 및 Quick Start (이 문서)
│   ├── PIPELINE_ARCHITECTURE.md   # 🔥 파이프라인 구조 상세 (AI 참조용)
│   └── CONFIG_GUIDE.md            # Config 파일 및 자동 실행 가이드
│
├── 🔧 Configuration/
│   ├── config_raw_data.py         # Raw 데이터 생성 설정
│   ├── config_pred.py             # 모델 학습/예측 설정
│   └── run_zeppelin.py            # Zeppelin 자동 실행 스크립트
│
└── 📊 Scala Config/
    ├── config_raw_data.py         # Raw 데이터 설정
    └── config_pred.py             # 예측 설정
```

## 🚀 Quick Start

### 1. 시간 범위 설정
`predict_ost_zpln.scala`의 **Paragraph 3**에서 시간 조건 변수 수정:

```scala
val sendMonth = "202512"              // 기준 월
val predictionDTSta = "20251101"      // Test 시작일
val predDT = "20251201"               // 예측 날짜
```

### 2. 작업 흐름 실행

#### 전체 파이프라인 실행
```
Paragraph 1 → Paragraph 32 순차 실행
```

#### 개별 작업 흐름 실행
- **Raw Data 생성**: P1-P14
- **Transformed Data 생성**: P15-P20
- **모델 학습**: P21-P30
- **서비스 예측**: P16 + P31-P32

## 📖 문서 가이드

### 문서 구성

| 문서 | 대상 | 내용 | 읽는 순서 |
|------|------|------|----------|
| **README.md** | 모두 | 프로젝트 개요, Quick Start | 1️⃣ 시작 |
| **PIPELINE_ARCHITECTURE.md** | AI/개발자 | 파이프라인 구조, 데이터 흐름, 작업 패턴 | 2️⃣ 구조 이해 |
| **CONFIG_GUIDE.md** | 운영자 | Config 파일, 자동 실행, Paragraph ID | 3️⃣ 실행 방법 |

### AI Assistant용
- **`PIPELINE_ARCHITECTURE.md`**: 코드 구조, 데이터 흐름, 작업 패턴 등 상세 참조 문서
  - 새 세션 시작 시 이 문서를 먼저 읽으면 코드 이해 가능
  - 코드 수정 시 참조해야 할 섹션 안내
  - Section 1: 전체 파이프라인 개요
  - Section 3: Paragraph 구조
  - Section 4: 데이터 흐름
  - Section 5: 작업 시 주의사항
  - Section 6: 일반적인 작업 패턴

### 개발자/운영자용
- **`CONFIG_GUIDE.md`**: Config 파일과 Paragraph ID 매핑
  - Python 자동 실행 스크립트 사용법
  - Paragraph ID 확인 및 관리
  - 작업 흐름별 실행 전략
  - 파라미터 사용법 및 트러블슈팅

## 🔑 핵심 개념

### 5대 작업 흐름
1. **Raw Data 생성** (P3-P14): Response data + User features + App usage → Train/Test raw
2. **Transformed Data 생성** (P15-P20): Raw data + Pipeline → Transformed data
3. **모델 학습** (P21-P30): Transformed data → Click/Gap/Utility 모델
4. **서비스 예측** (P16, P31-P32): Prediction data → Propensity score

### 시간 조건 변수 중앙 관리
- **Paragraph 3**에서 모든 시간 변수를 한 곳에서 관리
- 버전 일관성 자동 검증
- 각 Paragraph는 P3의 변수 참조

### 데이터 흐름
```
Response Data → Feature Join → Raw Data → Transformation → Trained Model → Prediction
```

## 🛠️ 일반적인 작업

### 시간 범위 변경
```scala
// Paragraph 3에서만 수정
val sendMonth = "202601"
val predictionDTSta = "20251201"
```

### 새로운 Feature 추가
1. P10: 데이터 로딩
2. P12: Feature join
3. P17: 컬럼 분류
4. P18: Pipeline 확인

### 버전 업그레이드
```scala
// Paragraph 3에서 통합 관리
val transformRawDataVersion = "11"
val transformedTrainSaveVersion = "11"
val modelTrainDataVersion = "11"
```

## ⚠️ 주의사항

1. **시간 변수는 P3에서만 수정**
2. **버전 일치 확인** (P3 실행 시 자동 검증)
3. **메모리 관리**: Suffix 배치 크기 조정
4. **캐시 정리**: 사용 완료된 DataFrame unpersist

## 🐛 트러블슈팅

### 데이터가 없을 때
- P3의 시간 범위 확인
- 저장/로딩 경로 일치 확인
- 버전 번호 일치 확인

### 메모리 부족 (OOM)
- Suffix 배치 크기 감소 (`transformSuffixGroupSize`, `predSuffixGroupSize`)
- 불필요한 캐시 제거
- Repartition 수 조정

### 성능 저하
- Broadcast join 확인
- Repartition으로 shuffle 최적화
- 캐싱 전략 검토

## 📊 주요 데이터 경로

| 데이터 | 경로 |
|--------|------|
| Response data | `aos/sto/response` |
| Raw Train | `aos/sto/trainDFRev${version}` |
| Raw Test | `aos/sto/testDFRev` |
| Transformed Train | `aos/sto/transformedTrainDFXDR${version}` |
| Propensity Score | `aos/sto/propensityScoreDF` |

## 📚 추가 문서

- **`PIPELINE_ARCHITECTURE.md`**: 전체 구조 상세 설명
- **코드 내 주석**: 각 Paragraph별 상세 설명

## 🤝 AI Assistant 사용 시

### 새로운 세션 시작 체크리스트

```
□ 1. PIPELINE_ARCHITECTURE.md 읽기
     - Section 1: 전체 개요 파악
     - Section 2: 핵심 설계 원칙 이해
     - Section 4: 데이터 흐름 추적

□ 2. 현재 작업 식별
     - 시간 범위 변경?         → P3 수정
     - Feature 추가?            → Section 5.2 참조
     - 버전 업그레이드?         → Section 6.2 참조
     - 모델 튜닝?               → P22-P26 참조

□ 3. 관련 Paragraph 확인
     - Section 3: Paragraph 구조
     - Section 4: 데이터 흐름도
     - Section 5: 주의사항

□ 4. 작업 수행
     - P3 시간 조건 변수 먼저 확인
     - 버전 일관성 검증 (P3 실행)
     - 해당 Paragraph 수정
     - 의존성 체크 (저장/로딩 경로)
```

### 일반 개발자 시작 체크리스트

```
□ 1. README.md 읽기 (Quick Start)
□ 2. config_*.py 파일 확인 (자동 실행 이해)
□ 3. CONFIG_GUIDE.md 읽기 (Paragraph ID 매핑)
□ 4. 테스트 실행 (작은 suffix로)
□ 5. PIPELINE_ARCHITECTURE.md 참조 (상세 이해)
```

---

## 📎 관련 링크

- [Zeppelin Notebook](http://150.6.14.94:30132/#/notebook/2MC68ADVY)
- Spark Cluster: `hdfs://scluster`
- Checkpoint Dir: `/user/g1110566/checkpoint`

---

**Last Updated**: 2026-01-22  
**Pipeline Version**: 1.0  
**Documentation Version**: 1.0
