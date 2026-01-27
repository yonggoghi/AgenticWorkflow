# jMetal Quick Start Guide

빠르게 jMetal 기반 최적화를 시작하는 가이드입니다.

## 1단계: JAR 다운로드

```bash
cd optimize_send_time
./download_jmetal.sh
```

> **Java 8 (Spark 3.1.3) 환경 주의**: Spark-shell이 Java 8이라면 jMetal은 6.x/5.10(=Java11)이 아니라 **Java 8 호환 jMetal 5.x(예: 5.6)** 를 사용해야 합니다.  
> `download_jmetal.sh`는 기본값이 Java8 호환 버전(현재 5.6)이며, 필요 시 `JMETAL_VERSION=6.1`로 오버라이드 가능합니다(단 Java 11+ 필요).

## 2단계: 환경 변수 설정

```bash
source setup_jmetal_env.sh
```

이제 `$JMETAL_JARS` 환경 변수가 설정되었습니다.

## 3단계: Spark Shell 실행

```bash
# jMetal만 사용
spark-shell --jars $JMETAL_JARS -i optimize_ost.scala

# OR-Tools와 함께 사용
spark-shell --jars $ALL_OPTIMIZER_JARS -i optimize_ost.scala
```

> **중요**: `-i optimize_ost.scala`로 이미 로드했다면, 같은 세션에서 다시 `:load optimize_ost.scala`를 실행하지 마세요.  
> (대형 파일/`case class` 재정의로 인해 spark-shell(Scala REPL)이 크래시할 수 있습니다.)
>
> - **방법 A(권장)**: 시작할 때 `-i optimize_ost.scala`를 쓰고, 이후에는 바로 함수 호출만 하기
> - **방법 B(개발용)**: `-i` 없이 spark-shell을 띄운 뒤 필요할 때만 `:load optimize_ost.scala`
>   - 재로딩이 필요하면 `:quit` 후 재시작(가장 안전) 또는 `:reset` 후 `:load`

## 4단계: 최적화 실행

```scala
import OptimizeSendTime._

// 데이터 로드
val dfAll = spark.read.parquet("aos/sto/propensityScoreDF").cache()
val df = dfAll.filter("svc_mgmt_num like '%00'")

// 용량 설정
val capacityPerHour = Map(
  9 -> 10000, 10 -> 10000, 11 -> 10000,
  12 -> 10000, 13 -> 10000, 14 -> 10000,
  15 -> 10000, 16 -> 10000, 17 -> 10000, 18 -> 10000
)

// NSGA-II 실행
val result = allocateUsersWithJMetalNSGAII(
  df = df,
  capacityPerHour = capacityPerHour,
  populationSize = 100,
  maxEvaluations = 25000
)

// 결과 확인
result.show()
result.groupBy("assigned_hour").agg(
  count("*").as("count"),
  sum("score").as("total_score"),
  avg("score").as("avg_score")
).orderBy("assigned_hour").show()
```

## 5단계: 자동 실행 스크립트 (NEW)

`load_and_test.scala`처럼 한 번에 로드+실행하려면 아래 스크립트를 사용하세요:

```bash
# (권장) stack을 올려 :load 크래시 완화
SPARK_SHELL_XSS=8m ./spark-shell-with-lib.sh -i optimize_ost.scala -i load_and_test_jmetal.scala -i load_and_test_jmetal.sc
```

## 알고리즘 선택

### NSGA-II (다양한 해 탐색)

```scala
val result = allocateUsersWithJMetalNSGAII(
  df = df,
  capacityPerHour = capacityPerHour,
  populationSize = 100,
  maxEvaluations = 25000
)
```

### MOEA/D (빠른 수렴)

```scala
val result = allocateUsersWithJMetalMOEAD(
  df = df,
  capacityPerHour = capacityPerHour,
  populationSize = 100,
  maxEvaluations = 25000
)
```

### Hybrid (jMetal + Greedy)

```scala
val result = allocateUsersHybridJMetal(
  df = df,
  capacityPerHour = capacityPerHour,
  algorithm = "NSGAII"  // 또는 "MOEAD"
)
```

### 대규모 배치 처리

```scala
val result = allocateLargeScaleJMetal(
  df = dfAll,  // 전체 데이터
  capacityPerHour = capacityPerHour,
  batchSize = 50000,
  algorithm = "NSGAII"
)
```

## 다른 알고리즘과 비교

```scala
// OR-Tools
val resultOR = allocateUsersWithHourlyCapacity(df, capacityPerHour)

// Simulated Annealing
val resultSA = allocateUsersWithSimulatedAnnealing(df, capacityPerHour)

// Greedy
val hours = Array(9,10,11,12,13,14,15,16,17,18)
val resultGreedy = allocateGreedySimple(df, hours, capacityPerHour)

// jMetal NSGA-II
val resultJMetal = allocateUsersWithJMetalNSGAII(df, capacityPerHour)

// 비교
println("OR-Tools score: " + resultOR.agg(sum("score")).first().getDouble(0))
println("SA score: " + resultSA.agg(sum("score")).first().getDouble(0))
println("Greedy score: " + resultGreedy.agg(sum("score")).first().getDouble(0))
println("jMetal score: " + resultJMetal.agg(sum("score")).first().getDouble(0))
```

## 문제 해결

### 환경 변수가 설정되지 않음

```bash
# 현재 쉘에서만 설정
source setup_jmetal_env.sh

# 영구적으로 설정 (선택사항)
echo 'source ~/workspace/AgenticWorkflow/optimize_send_time/setup_jmetal_env.sh' >> ~/.zshrc
source ~/.zshrc
```

### ClassNotFoundException

```bash
# 환경 변수 확인
echo $JMETAL_JARS

# JAR 파일 존재 확인
ls -l lib/*.jar

# 재다운로드
./download_jmetal.sh
source setup_jmetal_env.sh
```

### OutOfMemoryError

```bash
# 메모리 증가
spark-shell --driver-memory 8g --executor-memory 8g --jars $JMETAL_JARS

# 또는 배치 크기 감소
val result = allocateLargeScaleJMetal(df, capacityPerHour, batchSize = 20000)
```

## 다음 단계

- 상세 문서: `JMETAL_SETUP.md`
- 예제 코드: `example_jmetal.scala`
- 파라미터 튜닝 가이드: `JMETAL_SETUP.md` 참조
