# data_transformation.scala 실행 경고 메시지 가이드

> 성공 실행 중 발생한 경고 메시지 분류 및 대응 방법

---

## 요약

| 경고 유형 | 심각도 | 원인 | 즉각 조치 필요 |
|-----------|--------|------|----------------|
| `Writing block rdd_X to disk` | ⚠️ 주의 | Executor 메모리 부족 → 디스크 spill | 권장 (executor 증설) |
| `MemoryStore: N blocks selected for dropping` | ⚠️ 주의 | Storage 메모리 한계 초과 | 권장 |
| `Exception in task: Java heap space` | ⚠️ 주의 | 일부 task heap OOM (재시도 성공) | 권장 |
| `Container from a bad node: Exit status: 52` | ℹ️ 정보 | 불량 노드에서 launch 실패 (자동 제외) | 불필요 |
| `broadcast_X written to disk` | ℹ️ 정보 | 메모리 부족 시 broadcast 디스크 spill | 권장 |
| `Block rdd_X could not be removed` | ℹ️ 정보 | 이미 evict된 블록 제거 시도 (race condition) | 불필요 |
| `ShutdownHook timeout / InterruptedException` | ✅ 무시 | 정상 종료 과정 | 불필요 |

---

## 경고별 상세 설명

### 1. `Writing block rdd_X to disk` / `MemoryStore: N blocks selected for dropping`

```
INFO BlockManager: Writing block rdd_25_66 to disk
INFO MemoryStore: 5 blocks selected for dropping (726.5 MiB bytes)
```

**원인**: Executor의 Storage 메모리(MEMORY_AND_DISK_SER로 cache한 데이터)가 꽉 차서
남은 블록을 디스크로 spill하는 정상적인 Spark 동작이나,
디스크 I/O 증가 → 전체 처리 속도 저하로 이어짐.

**현재 설정 기준 per-executor storage 메모리**:
```
EXECUTOR_MEMORY = 30g
spark.memory.fraction = 0.8       → unified memory = 24g
spark.memory.storageFraction = 0.5 → storage = 12g
```
executor 30개로 10.7M 레코드를 처리하면 executor당 약 350K 레코드 + Word2Vec 모델 broadcast 수신.

**권장 조치**: [executor 증설 섹션](#권장-설정-executor-30--60) 참고

---

### 2. `Exception in task: Java heap space`

```
ERROR Executor: Exception in task 441.0 in stage 40.0 (TID 12728): Java heap space
```

**원인**: Word2Vec transform 단계에서 executor JVM heap이 부족한 task 발생.
`spark.task.maxFailures=4` 설정으로 자동 재시도 후 성공.

**특이사항**: EXECUTOR_MEMORY 대부분이 Spark unified memory로 할당되고
JVM native heap에 여유가 없을 때 발생.

**현재 JVM 옵션**:
```
-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35
```
G1GC 사용 중이므로 GC 자체는 최적화되어 있으나, heap 절대량이 부족한 상황.

**권장 조치**: executor 증설 시 per-executor 데이터 감소 → 자연히 해소됨

---

### 3. `Container from a bad node: Exit status: 52`

```
WARN TaskSetManager: Lost task 526.0 in stage 40.0 ...
  Container from a bad node: container_e75_...
  Exit status: 52
  Exception message: Launch container failed
```

**원인**: YARN 클러스터의 불안정한 노드에서 컨테이너 시작 실패.
Exit code 52 = YARN NodeManager의 컨테이너 실행 실패 (노드 문제).

**현재 설정**:
```
spark.excludeOnFailure.enabled=true
spark.excludeOnFailure.timeout=1h
spark.excludeOnFailure.task.maxTaskAttemptsPerNode=2
spark.excludeOnFailure.stage.maxFailedTasksPerExecutor=2
```

**판단**: 이미 `excludeOnFailure`가 활성화되어 있어 문제 노드는 자동으로 1시간 제외됨.
**추가 조치 불필요** — 운영 클러스터의 정상적인 transient failure.

---

### 4. `broadcast_X written to disk`

```
INFO BlockManager: Writing block broadcast_7_piece0 to disk
INFO BlockManager: Writing block broadcast_24_piece0 to disk
```

**원인**: Word2Vec 모델 등 broadcast 데이터가 Storage 메모리에서 evict되어 디스크 저장.
디스크에 저장된 broadcast는 필요 시 재로드 가능하므로 correctness 문제 없음.

**판단**: 경고 아님. 메모리 압박 신호로 참고만 하면 됨.

---

### 5. `Block rdd_X could not be removed as it was not found`

```
WARN BlockManager: Block rdd_139_441 could not be removed as it was not found on disk or in memory
```

**원인**: `.unpersist()`를 호출했을 때 해당 블록이 이미 memory에서 evict되어
디스크에도 없는 상태 (GC나 다른 eviction으로 이미 사라진 상태).

**판단**: 무해함. Spark이 이미 없는 블록을 제거하려 했을 뿐, 데이터 손실 없음.

---

### 6. `ShutdownHook timeout` / `java.lang.InterruptedException`

```
WARN ShutdownHook '$anon$2' timeout. java.util.concurrent.TimeoutException
ERROR Utils: Uncaught exception in thread shutdown-hook-0, java.lang.InterruptedException
```

**원인**: Spark 작업 완료 후 cleanup 과정에서 발생하는 정상적인 메시지.
JVM 종료 시 shutdown hook이 타임아웃 이내에 끝나지 못할 때 강제 중단됨.

**판단**: 완전히 무시해도 됨. 결과 데이터는 이미 HDFS에 저장된 후.

---

## 권장 설정: executor 30 → 60

`run_spark_shell.sh` 의 `NUM_EXECUTORS` 를 **30 → 60** 으로 변경:

```bash
# 변경 전
NUM_EXECUTORS=30
EXECUTOR_MEMORY="30g"   # per-executor storage ≈ 12g

# 변경 후
NUM_EXECUTORS=60        # ← 2배로 증설
EXECUTOR_MEMORY="30g"   # 그대로 유지
```

**효과**:

| 항목 | 30 executors | 60 executors |
|------|-------------|-------------|
| 총 cores | 150 | 300 |
| per-executor 데이터 | ~350K rows | ~175K rows |
| per-executor storage | 12g (30g × 0.8 × 0.5) | 12g (동일) |
| 디스크 spill 가능성 | 높음 | 낮음 |
| heap OOM 가능성 | 있음 | 낮음 |
| Word2Vec broadcast 부담 | 큼 | 분산됨 |

> **executor를 늘려도 per-executor 메모리는 동일하지만,
> 각 executor가 처리하는 데이터량이 절반으로 줄어 spill이 크게 감소함.**

### 변경 방법

`run_spark_shell.sh` line 35:

```bash
NUM_EXECUTORS=60              # 30→60: 디스크 spill / heap OOM 감소
```

---

## 추가 튜닝 옵션 (선택)

경고가 계속될 경우 아래 옵션 추가 검토:

```bash
# Shuffle 관련 개선 (shuffle read 중 heap 절약)
--conf spark.reducer.maxSizeInFlight=96m \
--conf spark.shuffle.file.buffer=1m \

# Spill 임계값 조정 (더 일찍 디스크로 내려 heap 보호)
--conf spark.shuffle.spill.numElementsForceSpillThreshold=100000 \

# Speculation: 느린 task 재실행 (bad node 영향 감소)
--conf spark.speculation=true \
--conf spark.speculation.multiplier=2 \
```

---

## 실행 단계별 메모리 압박 구간

```
[Stage 30-35] rawDF 로드 + Join    → broadcast join (200MB 이하 테이블)
[Stage 38-40] Click transform      ← ⚠️ Word2Vec 모델 broadcast + 10.7M 레코드 변환
[Stage 42-45] Gap transform        ← ⚠️ 두 번째 대형 transform
[Stage 50+]   XGBoost 학습         → executor 간 데이터 교환 (shuffle)
```

Word2Vec transform (Stage 40 전후)에서 `Java heap space` 및 디스크 spill이
집중 발생하므로 executor 증설의 효과가 가장 큰 구간임.
