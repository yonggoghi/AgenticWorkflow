# 🚀 빠른 시작 가이드

## 5분 안에 시작하기

### 1️⃣ 환경 설정 (최초 1회)

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/optimize_send_time

# Spark 환경 설치
./setup_spark_env.sh
source ~/.zshrc

# OR-Tools JAR 다운로드
./setup_ortools_jars.sh
```

### 2️⃣ 샘플 데이터 생성

```bash
# 방법 1: 간단한 방법 (추천)
./generate_data_simple.sh

# 방법 2: 사용자 수 지정
./generate_data_simple.sh 10000  # 10,000명

# 방법 3: 대화형 메뉴
./generate_sample_data.sh
```

### 3️⃣ 테스트 실행

```bash
# 방법 1: 자동 로드 및 테스트
spark-shell --driver-memory 4g -i load_and_test.scala

# 방법 2: 수동 로드 (권장 - Interactive)
./run_interactive.sh
# Spark Shell에서:
scala> :load optimize_ost.scala
scala> import OptimizeSendTime._

# 방법 3: 기존 방식 (deprecated)
spark-shell --driver-memory 4g -i optimize_ost.scala
```

> **중요**: `-i optimize_ost.scala`로 시작한 세션에서는 같은 파일을 다시 `:load optimize_ost.scala`로 실행하지 마세요.  
> 동일 정의 재로딩으로 spark-shell이 크래시할 수 있습니다. 재로딩이 필요하면 `:quit` 후 재시작이 가장 안전합니다.

### 4️⃣ Interactive 사용법

#### Spark Shell에서 수동 로드
```bash
# 1. Spark Shell 시작
./run_interactive.sh

# 또는 직접 실행
source ortools_env.sh
spark-shell --jars "$ORTOOLS_JARS" --driver-memory 4g
```

```scala
// 2. Spark Shell 내부에서
scala> :load optimize_ost.scala
// ✓ 로드 완료 메시지 표시됨

scala> import OptimizeSendTime._

// 3. 데이터 로드
scala> val dfAll = spark.read.parquet("aos/sto/propensityScoreDF").cache()
scala> val df = dfAll.limit(1000)

// 4. 용량 설정
scala> val capacity = Map(
     | 9 -> 100, 10 -> 100, 11 -> 100, 12 -> 100, 13 -> 100,
     | 14 -> 100, 15 -> 100, 16 -> 100, 17 -> 100, 18 -> 100
     | )

// 5. Greedy 실행
scala> val result = allocateGreedySimple(df, Array(9,10,11,12,13,14,15,16,17,18), capacity)

// 6. 결과 확인
scala> result.groupBy("assigned_hour").count().orderBy("assigned_hour").show()
```

#### 자동 테스트 스크립트
```bash
# 모든 것을 자동으로 실행
spark-shell --driver-memory 4g -i load_and_test.scala
```

### 5️⃣ 다양한 알고리즘 테스트

```scala
// 이미 로드된 데이터 사용
val df = dfAll.limit(10000)

// 용량 설정
val capacityPerHourMap = Map(
  9 -> 1000, 10 -> 1000, 11 -> 1000, 12 -> 1000,
  13 -> 1000, 14 -> 1000, 15 -> 1000, 16 -> 1000,
  17 -> 1000, 18 -> 1000
)

// SA 최적화 실행 (빠른 테스트)
import OptimizeSendTime._
val result = allocateUsersWithSimulatedAnnealing(
  df = df,
  capacityPerHour = capacityPerHourMap,
  maxIterations = 10000,  // 빠른 테스트를 위해 줄임
  initialTemperature = 100.0,
  coolingRate = 0.99,
  batchSize = 500000
)

// 결과 확인
result.show()
result.groupBy("assigned_hour").count().orderBy("assigned_hour").show()
```

## 🎯 주요 함수

### 1. Greedy 할당 (가장 빠름)
```scala
allocateGreedySimple(df, hours, capacityMap)
```

### 2. OR-Tools 최적화 (정확함)
```scala
allocateUsersWithHourlyCapacity(df, capacityMap, timeLimit = 300)
```

### 3. Simulated Annealing (균형)
```scala
allocateUsersWithSimulatedAnnealing(df, capacityMap, maxIterations = 100000)
```

### 4. Hybrid (OR-Tools + Greedy)
```scala
allocateUsersHybrid(df, capacityMap)
```

### 5. 대규모 배치 처리
```scala
allocateLargeScaleHybrid(df, capacityMap, batchSize = 500000)
```

## 📊 성능 비교

| 방법 | 속도 | 품질 | 메모리 | 추천 용도 |
|------|------|------|--------|-----------|
| Greedy | ⚡⚡⚡ | ⭐⭐ | 적음 | 빠른 테스트 |
| OR-Tools | ⚡ | ⭐⭐⭐⭐⭐ | 많음 | 소규모 최적화 |
| SA | ⚡⚡ | ⭐⭐⭐⭐ | 중간 | 균형 잡힌 선택 |
| Hybrid | ⚡⚡ | ⭐⭐⭐⭐ | 중간 | 프로덕션 |
| Batch | ⚡ | ⭐⭐⭐⭐ | 적음 | 대규모 데이터 |

## 🔧 문제 해결

### ❌ "Java not found"
```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
source ~/.zshrc
```

### ❌ "Spark not found"
```bash
export SPARK_HOME=/Users/yongwook/spark-local/spark-3.1.3-bin-hadoop3.2
export PATH=$SPARK_HOME/bin:$PATH
source ~/.zshrc
```

### ❌ "Out of Memory"
```bash
# 메모리 증가
spark-shell --driver-memory 8g --executor-memory 8g -i optimize_ost.scala

# 또는 데이터 크기 줄이기
val df = dfAll.sample(0.1)  // 10% 샘플링
```

## 📌 팁

1. **처음 실행**: Greedy로 시작해서 빠르게 테스트
2. **메모리 부족**: batchSize를 줄이거나 데이터 샘플링
3. **느린 실행**: maxIterations를 줄이거나 preprocessing 활성화
4. **최고 품질**: OR-Tools 사용 (소규모 데이터만)
5. **대규모 데이터**: Batch 처리 방식 사용

## 🎓 학습 순서

1. ✅ 환경 설정 확인
2. ✅ 샘플 데이터로 Greedy 테스트
3. ✅ 소규모 실제 데이터로 SA 테스트
4. ✅ 파라미터 튜닝 (maxIterations, coolingRate 등)
5. ✅ 전체 데이터로 Batch 처리

## 💡 다음 단계

- `INSTALLATION_GUIDE.md`: 상세 설치 가이드
- `TROUBLESHOOTING.md`: ⭐ 문제 해결 가이드
- `LINUX_DEPLOYMENT_GUIDE.md`: 리눅스 서버 배포
- `DATA_GENERATION_GUIDE.md`: 데이터 생성 가이드
- `README_SBT.md`: SBT 프로젝트 가이드
- Spark UI: http://localhost:4040 (실행 중일 때)

## 🆘 문제가 있나요?

**TROUBLESHOOTING.md**를 참고하세요! 다음 내용을 포함합니다:
- ✅ `$ORTOOLS_JARS` 변수 문제
- ✅ Out of Memory 해결
- ✅ JAR 파일 찾을 수 없음
- ✅ 환경 변수 설정 문제
- ✅ 데이터 경로 문제
- ✅ 기타 일반적인 오류
