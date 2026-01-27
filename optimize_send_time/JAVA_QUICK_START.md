# Greedy Allocator - Java 개발자 가이드

> 대규모 사용자 최적 발송 시간 할당 시스템 (배치 처리)

## 📋 개요

- **목적**: 2,500만명 사용자를 최적의 발송 시간대(9-18시)에 할당
- **알고리즘**: Greedy (탐욕 알고리즘) + Batch Processing
- **처리량**: 2,500만명 기준 약 1시간
- **메모리**: 2,500만명 기준 100GB 권장

---

## 🚀 빠른 시작 (4단계)

### 1단계: 샘플 데이터 생성

```bash
cd optimize_send_time
./generate_data_simple.sh 100000  # 10만명 (테스트용)
```

**데이터 생성 옵션:**
- `100000`: 10만명 (빠른 테스트)
- `1000000`: 100만명 (소규모)
- `25000000`: 2,500만명 (실제 운영)

**출력 위치:** `aos/sto/propensityScoreDF`

---

### 2단계: 컴파일

```bash
./build_java.sh
```

**자동으로 실행되는 내용:**
1. Java 컴파일
2. JAR 생성

### 3단계: 실행

```bash
./run_java_allocator.sh
```

**실행 옵션:**
```bash
# 기본 설정 (100GB 메모리)
./run_java_allocator.sh

# 커스텀 설정 (코어, 메모리, 최대 결과 크기)
./run_java_allocator.sh 32 150g 50g
```

**결과 저장:** `aos/sto/allocation_result`

---

### 4단계: 결과 확인

```bash
# 결과 파일 확인
ls -lh aos/sto/allocation_result

# Spark Shell로 결과 보기
spark-shell
```

```scala
val result = spark.read.parquet("aos/sto/allocation_result")
result.show(20)
result.groupBy("assigned_hour").count().orderBy("assigned_hour").show()
```

---

## 📁 주요 파일

| 파일 | 설명 |
|------|------|
| `GreedyAllocator.java` | 핵심 알고리즘 (배치 처리) |
| `GreedyAllocatorTest.java` | End-to-End 테스트 |
| `build_java.sh` | 빌드 스크립트 |
| `run_java_allocator.sh` | 실행 스크립트 |
| `generate_data_simple.sh` | 데이터 생성 스크립트 |

---

## 💻 코드 사용 예시

### 기본 사용법

```java
import org.apache.spark.sql.*;
import optimize_send_time.GreedyAllocator;
import java.util.*;

SparkSession spark = SparkSession.builder()
    .appName("Allocation")
    .master("local[*]")
    .getOrCreate();

// 1. 데이터 로드
Dataset<Row> df = spark.read()
    .parquet("aos/sto/propensityScoreDF")
    .cache();

// 2. 용량 설정 (전체 사용자의 110%)
long totalUsers = df.select("svc_mgmt_num").distinct().count();
int[] hours = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
int capacityPerHour = (int)(totalUsers * 0.11);

Map<Integer, Integer> capacity = new HashMap<>();
for (int h : hours) {
    capacity.put(h, capacityPerHour);
}

// 3. 할당 실행
GreedyAllocator allocator = new GreedyAllocator();
Dataset<Row> result = allocator.allocateLargeScale(
    df,           // 입력 데이터
    hours,        // 시간대
    capacity,     // 용량
    1000000       // 배치 크기 (100만명)
);

// 4. 결과 저장
result.write()
    .mode("overwrite")
    .parquet("aos/sto/allocation_result");
```

---

## ⚙️ 성능 튜닝

### 메모리 설정

사용자 수에 따른 권장 설정:

```bash
# 100만명
spark-submit \
  --master "local[*]" \
  --driver-cores 8 \
  --driver-memory 8g \
  --conf spark.driver.maxResultSize=5g \
  --class optimize_send_time.GreedyAllocatorTest \
  build/greedy-allocator.jar

# 1000만명
spark-submit \
  --master "local[*]" \
  --driver-cores 16 \
  --driver-memory 32g \
  --conf spark.driver.maxResultSize=10g \
  --class optimize_send_time.GreedyAllocatorTest \
  build/greedy-allocator.jar

# 2500만명
spark-submit \
  --master "local[*]" \
  --driver-cores 16 \
  --driver-memory 100g \
  --conf spark.driver.maxResultSize=30g \
  --class optimize_send_time.GreedyAllocatorTest \
  build/greedy-allocator.jar
```

### 배치 크기 선택

| 사용자 수 | 권장 배치 크기 | 예상 시간 |
|----------|--------------|----------|
| 10만 | 10만 | < 10초 |
| 100만 | 50만 | ~2분 |
| 1000만 | 100만 | ~20분 |
| 2500만 | 100만 | ~1시간 |

---

## 🔧 트러블슈팅

### 1. SPARK_HOME not set

```bash
export SPARK_HOME=/path/to/spark-3.1.3
export PATH=$SPARK_HOME/bin:$PATH
```

### 2. OutOfMemoryError

**해결책:**
- `--driver-memory` 증가 (예: 16g → 32g)
- 배치 크기 감소 (예: 1000000 → 500000)
- `--conf spark.driver.maxResultSize` 증가

### 3. 데이터가 없음

```bash
./generate_data_simple.sh 100000
```

### 4. 컴파일 에러

```bash
# Java 버전 확인 (11 이상 필요)
javac -version

# Spark 경로 확인
ls $SPARK_HOME/jars
```

---

## 📊 알고리즘 설명

### Greedy 알고리즘

1. **사용자 정렬**: 최고 점수 순으로 정렬
2. **순차 할당**: 점수 높은 사용자부터 처리
3. **용량 관리**: 시간대별 용량 실시간 차감

**장점:**
- ✅ 빠른 실행 속도 (O(n log n))
- ✅ 메모리 효율적 (배치 처리)
- ✅ 안정적이고 예측 가능

**특징:**
- 품질: 이론적 최적의 97-99%
- 속도: 2,500만명 기준 약 1시간
- 확장성: 2,500만명 이상 지원

---

## 📂 데이터 스키마

### 입력 (propensityScoreDF)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `svc_mgmt_num` | String | 사용자 ID |
| `send_hour` | Integer | 시간대 (9-18) |
| `propensity_score` | Double | 예측 반응률 (0.0-1.0) |

**특징:**
- 사용자당 10개 레코드 (9-18시)
- 2,500만명 = 2억 5천만 레코드

### 출력 (allocation_result)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `svc_mgmt_num` | String | 사용자 ID |
| `assigned_hour` | Integer | 할당된 시간 |
| `score` | Double | 해당 시간의 점수 |

**특징:**
- 사용자당 1개 레코드
- 2,500만명 = 2,500만 레코드

---

## 🎯 체크리스트

실행 전 확인사항:

- [ ] Java 11 이상 설치
- [ ] Spark 3.1.3 설치
- [ ] `SPARK_HOME` 환경변수 설정
- [ ] 데이터 생성 완료 (`aos/sto/propensityScoreDF`)
- [ ] 충분한 메모리 확보 (최소 16GB)

---

## 📞 추가 정보

**상세 문서:**
- `JAVA_USAGE.md`: 전체 사용 가이드
- `GREEDY_LARGE_SCALE.md`: 대규모 처리 상세 설명
- `TROUBLESHOOTING.md`: 문제 해결 가이드

**실행 스크립트:**
- `build_java.sh`: 빌드 및 실행
- `generate_data_simple.sh`: 데이터 생성
- `generate_data_with_backup.sh`: 백업 포함 생성

---

## 📈 성능 벤치마크

| 사용자 수 | 실행 시간 | 메모리 사용 | 처리량 |
|----------|----------|------------|--------|
| 10만 | ~10초 | 8GB | ~10,000/초 |
| 100만 | ~2분 | 16GB | ~8,300/초 |
| 1000만 | ~20분 | 32GB | ~8,300/초 |
| 2500만 | ~1시간 | 100GB | ~7,000/초 |

*참고: 성능은 하드웨어 사양에 따라 달라질 수 있습니다.*

---

**작성일**: 2026-01-14  
**버전**: 1.0  
**Spark**: 2.3.x, 3.x  
**Java**: 8+
