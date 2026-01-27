# jMetal 설정 가이드

## 개요

`optimize_ost.scala`에 jMetal 기반 다목적 최적화 함수가 추가되었습니다. jMetal은 Java 기반의 Multi-Objective Optimization 프레임워크로, NSGA-II, MOEA/D 등의 알고리즘을 제공합니다.

## 주요 기능

### 다목적 최적화 목표

1. **목적 1: 총 propensity score 최대화**
   - 사용자별 최적 시간대 선택으로 전체 응답률 극대화

2. **목적 2: 시간대별 균등 분배 (부하 분산)**
   - 시간대별 사용자 수의 표준편차 최소화
   - 시스템 부하 균등 분산

### 제약 조건

- 시간대별 용량 제한 (capacityPerHour)
- 각 사용자는 사용 가능한 시간대 중에서만 선택

## 의존성 추가

### 방법 1: Maven/SBT 사용

**build.sbt에 추가:**

```scala
name := "OptimizeSendTime"
version := "1.0"
scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.3.0" % "provided",
  "com.google.ortools" % "ortools-java" % "9.4.1874",
  "org.uma.jmetal" % "jmetal-core" % "6.1",
  "org.uma.jmetal" % "jmetal-algorithm" % "6.1",
  "org.uma.jmetal" % "jmetal-problem" % "6.1"
)
```

**Maven pom.xml에 추가:**

```xml
<dependencies>
  <!-- jMetal dependencies -->
  <dependency>
    <groupId>org.uma.jmetal</groupId>
    <artifactId>jmetal-core</artifactId>
    <version>6.1</version>
  </dependency>
  <dependency>
    <groupId>org.uma.jmetal</groupId>
    <artifactId>jmetal-algorithm</artifactId>
    <version>6.1</version>
  </dependency>
  <dependency>
    <groupId>org.uma.jmetal</groupId>
    <artifactId>jmetal-problem</artifactId>
    <version>6.1</version>
  </dependency>
</dependencies>
```

### 방법 2: JAR 파일 직접 다운로드 (권장)

**자동 다운로드 스크립트 사용:**

```bash
# 현재 디렉토리에서 실행
./download_jmetal.sh
```

이 스크립트는 다음을 수행합니다:
- jMetal JAR 파일들을 `lib/` 디렉토리에 다운로드
- 편의 스크립트 생성
- 환경 변수 설정 방법 안내

**환경 변수 설정 (권장):**

```bash
# 환경 변수 설정
source setup_jmetal_env.sh

# 이제 $JMETAL_JARS 사용 가능
spark-shell --jars $JMETAL_JARS -i optimize_ost.scala

# OR-Tools와 함께 사용
spark-shell --jars $ORTOOLS_JARS,$JMETAL_JARS -i optimize_ost.scala

# 또는 자동으로 결합된 변수 사용
spark-shell --jars $ALL_OPTIMIZER_JARS -i optimize_ost.scala
```

> **중요**: 위처럼 `-i optimize_ost.scala`로 시작했다면, 같은 세션에서 `:load optimize_ost.scala`를 다시 실행하지 마세요.  
> 동일한 `case class`/`object` 재정의가 발생하면서 spark-shell(Scala REPL)이 크래시할 수 있습니다.
>
> - **개발 중 파일을 자주 다시 읽고 싶으면**: `spark-shell --jars ...`로만 시작하고 필요할 때 `:load optimize_ost.scala`
> - **재로딩이 필요하면**: 가장 안전하게 `:quit` 후 재시작(또는 `:reset` 후 `:load`)

**수동 다운로드:**

```bash
# jMetal JAR 다운로드
wget https://repo1.maven.org/maven2/org/uma/jmetal/jmetal-core/6.1/jmetal-core-6.1.jar
wget https://repo1.maven.org/maven2/org/uma/jmetal/jmetal-algorithm/6.1/jmetal-algorithm-6.1.jar
wget https://repo1.maven.org/maven2/org/uma/jmetal/jmetal-problem/6.1/jmetal-problem-6.1.jar

# spark-shell 실행 시 JAR 포함
spark-shell --jars jmetal-core-6.1.jar,jmetal-algorithm-6.1.jar,jmetal-problem-6.1.jar
```

## 환경 설정

### 쉘 프로파일에 추가 (선택사항)

매번 `source setup_jmetal_env.sh`를 실행하지 않으려면, 쉘 설정 파일에 추가하세요:

**Bash (~/.bashrc 또는 ~/.bash_profile):**

```bash
# jMetal 환경 변수
if [ -f ~/workspace/AgenticWorkflow/optimize_send_time/setup_jmetal_env.sh ]; then
    source ~/workspace/AgenticWorkflow/optimize_send_time/setup_jmetal_env.sh > /dev/null 2>&1
fi
```

**Zsh (~/.zshrc):**

```zsh
# jMetal 환경 변수
if [ -f ~/workspace/AgenticWorkflow/optimize_send_time/setup_jmetal_env.sh ]; then
    source ~/workspace/AgenticWorkflow/optimize_send_time/setup_jmetal_env.sh > /dev/null 2>&1
fi
```

### 환경 변수 확인

```bash
# 설정 확인
echo $JMETAL_JARS

# OR-Tools와 함께 설정되었는지 확인
echo $ALL_OPTIMIZER_JARS
```

## 사용 예제

### 0. Spark Shell 시작

```bash
# 환경 변수 설정
source setup_jmetal_env.sh

# Spark shell 시작 (jMetal만 사용)
spark-shell --jars $JMETAL_JARS -i optimize_ost.scala

# 또는 OR-Tools와 함께 사용
spark-shell --jars $ALL_OPTIMIZER_JARS -i optimize_ost.scala
```

### 1. NSGA-II 알고리즘 사용

```scala
// 데이터 로드
val dfAll = spark.read.parquet("aos/sto/propensityScoreDF").cache()
val df = dfAll.filter("svc_mgmt_num like '%00'")

// 시간대별 용량 설정
val capacityPerHour = Map(
  9 -> 10000, 10 -> 10000, 11 -> 10000,
  12 -> 10000, 13 -> 10000, 14 -> 10000,
  15 -> 10000, 16 -> 10000, 17 -> 10000, 18 -> 10000
)

// NSGA-II 실행
import OptimizeSendTime._
val result = allocateUsersWithJMetalNSGAII(
  df = df,
  capacityPerHour = capacityPerHour,
  populationSize = 100,           // 개체군 크기
  maxEvaluations = 25000,         // 최대 평가 횟수
  crossoverProbability = 0.9,     // 교차 확률
  mutationProbability = 0.1       // 돌연변이 확률
)

// 결과 확인
result.show(20, false)
result.groupBy("assigned_hour").agg(
  count("*").as("count"),
  sum("score").as("total_score"),
  avg("score").as("avg_score")
).orderBy("assigned_hour").show()
```

### 2. MOEA/D 알고리즘 사용

```scala
// MOEA/D 실행 (분해 기반 접근)
val result = allocateUsersWithJMetalMOEAD(
  df = df,
  capacityPerHour = capacityPerHour,
  populationSize = 100,
  maxEvaluations = 25000,
  neighborhoodSize = 20  // 이웃 크기
)
```

### 3. Hybrid 방식 (jMetal + Greedy)

```scala
// jMetal로 대부분 할당하고, 남은 사용자는 Greedy로 할당
val result = allocateUsersHybridJMetal(
  df = df,
  capacityPerHour = capacityPerHour,
  algorithm = "NSGAII",  // "NSGAII" 또는 "MOEAD"
  populationSize = 100,
  maxEvaluations = 25000
)
```

### 4. 대규모 배치 처리

```scala
// 대용량 데이터를 배치로 나누어 처리
val result = allocateLargeScaleJMetal(
  df = dfAll,  // 전체 데이터
  capacityPerHour = capacityPerHour,
  batchSize = 50000,        // jMetal은 메모리 사용량이 크므로 작은 배치
  algorithm = "NSGAII",
  populationSize = 100,
  maxEvaluations = 25000
)
```

## 알고리즘 비교

| 알고리즘 | 특징 | 속도 | 품질 | 메모리 |
|---------|------|------|------|--------|
| **NSGA-II** | Pareto 기반, 다양한 해 탐색 | 중간 | 높음 | 높음 |
| **MOEA/D** | 분해 기반, 효율적 탐색 | 빠름 | 중간-높음 | 중간 |
| **OR-Tools** | 단일 목표, 정확한 최적해 | 느림 | 매우 높음 | 낮음 |
| **SA** | 단일 목표, 준최적해 | 빠름 | 중간 | 낮음 |
| **Greedy** | 휴리스틱, 빠른 할당 | 매우 빠름 | 낮음 | 매우 낮음 |

## 파라미터 튜닝 가이드

### NSGA-II

- **populationSize**: 100-200 (문제 복잡도에 따라)
- **maxEvaluations**: 25000-100000 (더 많을수록 품질 향상)
- **crossoverProbability**: 0.8-0.95
- **mutationProbability**: 0.05-0.2

### MOEA/D

- **populationSize**: 100-200
- **maxEvaluations**: 25000-100000
- **neighborhoodSize**: 10-30 (population의 10-30%)

## 성능 최적화 팁

1. **배치 크기 조정**: jMetal은 메모리 사용량이 크므로 배치 크기를 50,000 이하로 권장
2. **평가 횟수**: maxEvaluations를 늘리면 품질이 향상되지만 시간이 증가
3. **알고리즘 선택**: 
   - 다양한 trade-off 해가 필요하면 NSGA-II
   - 빠른 수렴이 필요하면 MOEA/D
4. **Hybrid 사용**: 큰 데이터셋은 Hybrid 방식으로 jMetal + Greedy 조합

## 주의사항

1. **메모리 사용량**: jMetal은 전체 개체군을 메모리에 유지하므로 큰 문제(>100K 사용자)는 배치 처리 필요
2. **실행 시간**: OR-Tools보다는 빠르지만 Greedy보다는 느림
3. **제약 조건**: 현재 구현은 soft constraint로, 일부 제약이 위반될 수 있음
4. **Pareto front**: 여러 해 중 하나를 선택해야 하므로, 현재는 총 점수 기준으로 자동 선택

## 문제 해결

### ClassNotFoundException 발생 시

```bash
# JAR 경로 확인
ls -l jmetal-*.jar

# spark-shell에 명시적으로 추가
spark-shell --jars /path/to/jmetal-core-6.1.jar,/path/to/jmetal-algorithm-6.1.jar,/path/to/jmetal-problem-6.1.jar
```

### OutOfMemoryError 발생 시

```bash
# 메모리 증가
spark-shell --driver-memory 8g --executor-memory 8g

# 또는 배치 크기 감소
val result = allocateLargeScaleJMetal(df, capacityPerHour, batchSize = 20000)
```

## 참고 자료

- [jMetal 공식 문서](https://jmetal.github.io/jMetal/)
- [NSGA-II 논문](https://ieeexplore.ieee.org/document/996017)
- [MOEA/D 논문](https://ieeexplore.ieee.org/document/4358754)
