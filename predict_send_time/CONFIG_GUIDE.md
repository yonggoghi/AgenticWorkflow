# Configuration Files Guide

> Python 설정 파일과 Zeppelin Scala 코드 간의 관계 및 자동 실행 가이드

## 📁 설정 파일 구조

```
predict_send_time/
├── config_raw_data.py      # Raw Data 생성 워크플로우 설정
├── config_pred.py          # 모델 학습 및 예측 워크플로우 설정
└── run_zeppelin.py         # Zeppelin Notebook 자동 실행 스크립트
```

---

## 1. 설정 파일 이해

### 1.1 `config_raw_data.py` - Raw Data 생성

**목적**: Raw training/test 데이터 생성 (작업 흐름 1-2)

**실행 대상**:
```python
# Pre Paragraphs (P1-P9): 준비 단계
PARAGRAPH_IDS_PRE = [
    "paragraph_1764658338256_686533166",  # P1: Imports and Configuration
    "paragraph_1764742922351_426209997",  # P2: Helper Functions
    "paragraph_1764742953919_436300403",  # P3: Date Range Configuration ⭐
    "paragraph_1764659911196_1763551717",  # P4: Response Data Loading
    "paragraph_1764641394585_598529380",  # P5: Response Data Filtering
    "paragraph_1764739202982_181479704",  # P6: User Feature Loading (MMKT)
    "paragraph_1764739017819_1458690185",  # P7: Train/Test Split
    "paragraph_1764738582669_1614068999",  # P8: Undersampling Ratio
    "paragraph_1764756027560_85739584",   # P9: Training Data Undersampling
]

# Main Paragraphs (P10-P14): Suffix별 배치 실행
PARAGRAPH_IDS = [
    "paragraph_1766323923540_1041552789",  # P10: App Usage Data Loading
    "paragraph_1767594403472_2124174124",  # P11: Historical Click Count
    "paragraph_1764755002817_1620624445",  # P12: Feature Integration
    "paragraph_1764832142136_413314670",   # P13: Data Type Conversion
    "paragraph_1766224516076_433149416",   # P14: Raw Data Persistence
]

# Suffix 배치 실행
PARAMS = [f"suffix:{hex(i)[2:]}" for i in range(12, 16)]  # c, d, e, f
```

**실행 플로우**:
1. PRE 단계: P1-P9 순차 실행 (1회)
2. MAIN 단계: P10-P14를 각 suffix(c, d, e, f)마다 실행 (4회 반복)

**사용 예시**:
```bash
python run_zeppelin.py --config config_raw_data
```

---

### 1.2 `config_pred.py` - 모델 학습 및 예측

**목적**: Transformed data 로딩 후 모델 학습 및 평가 (작업 흐름 4)

**실행 대상**:
```python
# Pre Paragraphs (P1-P2, P17, P21): 준비 및 데이터 로딩
PARAGRAPH_IDS_PRE = [
    "paragraph_1764658338256_686533166",  # P1: Imports and Configuration
    "paragraph_1764742922351_426209997",  # P2: Helper Functions
    "paragraph_1764833771372_1110341451",  # P17: Pipeline Parameters
    "paragraph_1765521446308_1651058139",  # P21: Load Transformed Data ⭐
]

# Main Paragraphs (P22, P24-P28): 모델 학습 및 평가
PARAGRAPH_IDS = [
    "paragraph_1764836200898_700489598",   # P22: Model Definitions
    "paragraph_1765789893517_1550413688",  # P24: Click Model Training
    "paragraph_1767010803374_275395458",   # P25: Gap Model Training
    "paragraph_1765764610094_1504595267",  # P26: Utility Model Training
    "paragraph_1765345345715_612147457",   # P27: Prediction on Test Set
    "paragraph_1764838154931_1623772564",  # P28: Performance Evaluation
]

# 파라미터 없음 (전체 데이터 사용)
PARAMS = []
```

**실행 플로우**:
1. PRE 단계: P1, P2, P17, P21 순차 실행 (1회)
2. MAIN 단계: P22, P24-P28 순차 실행 (1회)

**사용 예시**:
```bash
python run_zeppelin.py --config config_pred
```

---

## 2. Paragraph ID 매핑

### 2.1 ID 확인 방법

Zeppelin Notebook에서 각 Paragraph의 ID는 다음과 같이 확인:

```scala
// ===== Paragraph N: [제목] (ID: paragraph_XXXXX) =====
```

예시:
```scala
// ===== Paragraph 3: Date Range Configuration (ID: paragraph_1764742953919_436300403) =====
```
→ ID: `"paragraph_1764742953919_436300403"`

### 2.2 전체 Paragraph ID 매핑

| Paragraph | Title | ID | Config File |
|-----------|-------|----|----|
| P1 | Imports and Configuration | `paragraph_1764658338256_686533166` | Both |
| P2 | Helper Functions | `paragraph_1764742922351_426209997` | Both |
| **P3** | **Date Range Configuration** | `paragraph_1764742953919_436300403` | **raw_data** |
| P4 | Response Data Loading | `paragraph_1764659911196_1763551717` | raw_data |
| P5 | Response Data Filtering | `paragraph_1764641394585_598529380` | raw_data |
| P6 | User Feature Loading | `paragraph_1764739202982_181479704` | raw_data |
| P7 | Train/Test Split | `paragraph_1764739017819_1458690185` | raw_data |
| P8 | Undersampling Ratio | `paragraph_1764738582669_1614068999` | raw_data |
| P9 | Training Data Undersampling | `paragraph_1764756027560_85739584` | raw_data |
| P10 | App Usage Data Loading | `paragraph_1766323923540_1041552789` | raw_data |
| P11 | Historical Click Count | `paragraph_1767594403472_2124174124` | raw_data |
| P12 | Feature Integration | `paragraph_1764755002817_1620624445` | raw_data |
| P13 | Data Type Conversion | `paragraph_1764832142136_413314670` | raw_data |
| P14 | Raw Data Persistence | `paragraph_1766224516076_433149416` | raw_data |
| P15 | Raw Data Loading | `paragraph_1766392634024_1088239830` | Manual |
| P16 | Prediction Dataset Prep | `paragraph_1765765120629_645290475` | Manual |
| **P17** | **Pipeline Parameters** | `paragraph_1764833771372_1110341451` | **pred** |
| P18 | Pipeline Function | `paragraph_1765330122144_909170709` | Manual |
| P19 | Pipeline Transformation | `paragraph_1767353227961_983246072` | Manual |
| P20 | Save Transformed Data | `paragraph_1765520460775_2098641576` | Manual |
| **P21** | **Load Transformed Data** | `paragraph_1765521446308_1651058139` | **pred** |
| P22 | Model Definitions | `paragraph_1764836200898_700489598` | pred |
| P23 | XGBoost Constraints | `paragraph_1765939568349_1781513249` | Manual |
| P24 | Click Model Training | `paragraph_1765789893517_1550413688` | pred |
| P25 | Gap Model Training | `paragraph_1767010803374_275395458` | pred |
| P26 | Utility Model Training | `paragraph_1765764610094_1504595267` | pred |
| P27 | Model Prediction | `paragraph_1765345345715_612147457` | pred |
| P28 | Performance Evaluation | `paragraph_1764838154931_1623772564` | pred |
| P29 | Gap Model Evaluation | `paragraph_1767010293011_1290077245` | Manual |
| P30 | Regression Evaluation | `paragraph_1765786040626_1985577608` | Manual |
| P31 | Propensity Score Calc | `paragraph_1765768974381_910321724` | Manual |
| P32 | Propensity Score Loading | `paragraph_1767943423474_1143363402` | Manual |

---

## 3. 작업 흐름별 실행 전략

### 3.1 전체 파이프라인 실행 (처음부터 끝까지)

```bash
# Step 1: Raw Data 생성 (P1-P14, suffix별 배치)
python run_zeppelin.py --config config_raw_data

# Step 2: Transformed Data 생성 (P15-P20, 수동 실행)
# Zeppelin UI에서 P15-P20 실행

# Step 3: 모델 학습 및 평가 (P21-P28)
python run_zeppelin.py --config config_pred

# Step 4: 서비스 예측 (P31-P32, 수동 실행)
# Zeppelin UI에서 P16, P31-P32 실행
```

### 3.2 부분 실행 (특정 단계만)

#### Raw Data만 재생성
```bash
python run_zeppelin.py --config config_raw_data
```

#### 모델만 재학습 (데이터는 그대로)
```bash
python run_zeppelin.py --config config_pred
```

#### 특정 suffix만 처리
```python
# config_raw_data.py 수정
PARAMS = [f"suffix:0"]  # 0번 suffix만
```

---

## 4. 파라미터 사용법

### 4.1 Suffix 파라미터

**P10의 코드 참조**:
```scala
val smnSuffix = z.input("suffix", "0").toString
val smnCond = smnSuffix.split(",").map(c => s"svc_mgmt_num like '%${c}'").mkString(" or ")
```

**Config 설정**:
```python
# 단일 suffix
PARAMS = ["suffix:0"]

# 복수 suffix (한 번에 처리)
PARAMS = ["suffix:0,1,2,3"]

# 배치 실행 (각각 별도 실행)
PARAMS = [f"suffix:{i}" for i in range(16)]  # 0-f

# 범위 지정
PARAMS = [f"suffix:{hex(i)[2:]}" for i in range(12, 16)]  # c-f
```

### 4.2 추가 파라미터 예시

```python
# 복수 파라미터
PARAMS = [
    ["suffix:0", "month:202512"],
    ["suffix:1", "month:202512"],
]

# 코드에서 사용 (P3 등에서)
val targetMonth = z.input("month", "202512").toString
```

---

## 5. 자동 실행 스크립트 (`run_zeppelin.py`)

### 5.1 기본 사용법

```bash
# Raw data 생성
python run_zeppelin.py --config config_raw_data

# 모델 학습
python run_zeppelin.py --config config_pred

# 도움말
python run_zeppelin.py --help
```

### 5.2 Spark 재시작 옵션

```python
# config_*.py에서 설정
RESTART_SPARK_AT_START = True   # 시작 전 Spark 재시작
RESTART_SPARK_AT_END = True     # 완료 후 Spark 재시작
```

**사용 시나리오**:
- `START=True, END=False`: 메모리 정리 후 시작, 결과 유지
- `START=False, END=True`: 연속 실행, 완료 후 정리
- `START=True, END=True`: 독립적 실행, 전후 정리

---

## 6. 시간 조건 변경 시 워크플로우

### 6.1 시나리오: 2026년 1월 데이터로 변경

**Step 1**: Scala 코드에서 P3 수정
```scala
// predict_ost_zpln.scala - Paragraph 3
val sendMonth = "202601"              // 202512 → 202601
val featureMonth = "202512"           // 202511 → 202512
val predictionDTSta = "20251201"      // 20251101 → 20251201
val predictionDTEnd = "20260101"      // 20251201 → 20260101
val predDT = "20260101"               // 20251201 → 20260101
```

**Step 2**: Config 파일 확인 (변경 불필요)
```python
# config_raw_data.py - 그대로 사용
# Paragraph 3이 자동으로 새 시간 조건 사용
```

**Step 3**: 자동 실행
```bash
python run_zeppelin.py --config config_raw_data
```

### 6.2 버전 업그레이드 시

**Step 1**: P3에서 버전 변경
```scala
// Paragraph 3
val transformRawDataVersion = "11"        // 10 → 11
val transformedTrainSaveVersion = "11"
val modelTrainDataVersion = "11"
```

**Step 2**: Raw data 재생성
```bash
python run_zeppelin.py --config config_raw_data
```

**Step 3**: Transformed data 재생성 (수동)
- Zeppelin UI에서 P15-P20 실행

**Step 4**: 모델 재학습
```bash
python run_zeppelin.py --config config_pred
```

---

## 7. 트러블슈팅

### 7.1 Config 파일 관련

**문제**: Paragraph가 실행되지 않음
- [ ] Paragraph ID가 정확한가?
- [ ] Zeppelin 서버 주소가 올바른가?
- [ ] Notebook ID가 맞는가?

**문제**: Suffix 파라미터가 전달되지 않음
- [ ] PARAMS 형식이 올바른가? (`"suffix:0"`)
- [ ] P10의 `z.input("suffix", "0")` 코드가 있는가?

**문제**: PRE 단계는 성공했는데 MAIN이 실패
- [ ] P3의 시간 범위가 데이터와 일치하는가?
- [ ] Suffix별로 데이터가 존재하는가?

### 7.2 실행 순서 관련

**올바른 순서**:
1. `config_raw_data` → Raw data 생성
2. P15-P20 (수동) → Transformed data 생성
3. `config_pred` → 모델 학습

**잘못된 순서**:
- ❌ `config_pred`를 먼저 실행 (transformed data 없음)
- ❌ P15를 Raw data 생성 전에 실행 (raw data 없음)

---

## 8. 고급 설정

### 8.1 새로운 Config 파일 추가

**예시**: Transformed data 생성 자동화

```python
# config_transform.py
ZEPP_URL = "http://150.6.14.94:30132"
NOTEBOOK_ID = "2MC68ADVY"

PARAGRAPH_IDS_PRE = [
    "paragraph_1764658338256_686533166",  # P1
    "paragraph_1764742922351_426209997",  # P2
    "paragraph_1764742953919_436300403",  # P3
    "paragraph_1766392634024_1088239830",  # P15: Load Raw Data
]

PARAGRAPH_IDS = [
    "paragraph_1765765120629_645290475",  # P16: Prediction Dataset
    "paragraph_1764833771372_1110341451",  # P17: Pipeline Parameters
    "paragraph_1765330122144_909170709",  # P18: Pipeline Function
    "paragraph_1767353227961_983246072",  # P19: Pipeline Transformation
    "paragraph_1765520460775_2098641576",  # P20: Save Transformed Data
]

PARAMS = []
RESTART_SPARK_AT_START = True
RESTART_SPARK_AT_END = True
```

**사용**:
```bash
python run_zeppelin.py --config config_transform
```

### 8.2 Paragraph ID 자동 추출

```python
# 향후 개선: Notebook에서 Paragraph ID 자동 추출
import requests

def get_paragraph_ids(zepp_url, notebook_id):
    response = requests.get(f"{zepp_url}/api/notebook/{notebook_id}")
    notebook = response.json()
    return [p['id'] for p in notebook['body']['paragraphs']]
```

---

## 9. Quick Reference

### 자주 사용하는 명령어

```bash
# Raw data 생성 (전체)
python run_zeppelin.py --config config_raw_data

# Raw data 생성 (특정 suffix만, config 수정 필요)
# PARAMS = ["suffix:0"]

# 모델 학습
python run_zeppelin.py --config config_pred

# 로그 확인
tail -f zeppelin_execution.log
```

### Config 파일 구조

```python
ZEPP_URL = "서버주소"
NOTEBOOK_ID = "노트북ID"
PARAGRAPH_IDS_PRE = [...]  # 사전 실행 (1회)
PARAGRAPH_IDS = [...]       # 메인 실행 (PARAMS 반복)
PARAMS = [...]              # 파라미터 리스트
RESTART_SPARK_AT_START = True/False
RESTART_SPARK_AT_END = True/False
```

---

## 마지막 업데이트
- **날짜**: 2026-01-22
- **버전**: 1.0
- **다음 검토**: Config 구조 변경 시
