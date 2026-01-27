#!/bin/bash

# ===== 코드 분석 기반 최적화된 Spark 설정 =====
#
# 워크로드 특성:
# 1. 12M+ users 대상 MMS 광고 응답 예측 (Click, Response Time)
# 2. Multiple joins (8+ joins): MMKT + XDR + 기타 피처
# 3. XGBoost 모델 학습 (3개 모델: Click, Gap, Regression)
# 4. Token embedding 처리 (app_usage_token)
# 5. Suffix별 16번 반복 예측 (0-f)
# 6. 대량 캐싱 필요 (resDF, trainDF, transformedDF)
#
# 최적화 전략:
# - Join 최적화: Broadcast join threshold 증가, AQE 활성화
# - 메모리: 캐싱 및 XGBoost 학습을 위한 충분한 메모리
# - Shuffle: 대용량 join을 위한 적절한 파티션 수
# - Serialization: Kryo for faster data transfer
# ===================================================================

SPARK_MASTER="yarn"
DEPLOY_MODE="client"
APP_NAME="predict_ost"

# ===== 리소스 설정 (코드 복잡도 기반) =====
# 원본 Zeppelin: 100 executors × 4 cores × 30GB
# 
# 권장 조정:
# - 개발/테스트: 20-30 executors
# - 운영: 60-100 executors (데이터 크기에 따라)

DRIVER_CORES=8
DRIVER_MEMORY="50g"           # XGBoost 모델 학습 결과 수집용
EXECUTOR_CORES=5              # 4→5: XGBoost 병렬 처리 효율
EXECUTOR_MEMORY="35g"         # 36g→35g: YARN 44GB 제한 준수 (35+1+7=43GB)
NUM_EXECUTORS=30              # 100→80: 리소스 효율화

echo "=========================================="
echo "Starting Spark Shell (Optimized for ML Pipeline)"
echo "=========================================="
echo "Workload: MMS Campaign Response Prediction"
echo "- 12M+ users, Multi-model training (XGBoost)"
echo "- Multiple joins (8+), Token embedding"
echo "- Suffix-based batch prediction (16 iterations)"
echo "=========================================="
echo "Master: $SPARK_MASTER"
echo "App Name: $APP_NAME"
echo "Driver: ${DRIVER_CORES} cores, ${DRIVER_MEMORY} memory"
echo "Executor: ${EXECUTOR_CORES} cores, ${EXECUTOR_MEMORY} memory, ${NUM_EXECUTORS} instances"
echo "Total: $((NUM_EXECUTORS * EXECUTOR_CORES)) cores, $((NUM_EXECUTORS * ${EXECUTOR_MEMORY%g}))GB memory"
echo "=========================================="
echo ""

spark-shell \
  --master $SPARK_MASTER \
  --deploy-mode $DEPLOY_MODE \
  --name "$APP_NAME" \
  --driver-cores $DRIVER_CORES \
  --driver-memory $DRIVER_MEMORY \
  --executor-cores $EXECUTOR_CORES \
  --executor-memory $EXECUTOR_MEMORY \
  --num-executors $NUM_EXECUTORS \
  \
  `# ===== Join 최적화 =====` \
  --conf spark.sql.autoBroadcastJoinThreshold=209715200 \
  `# 100MB→200MB: MMKT/XDR 피처 테이블 broadcast` \
  --conf spark.sql.broadcastTimeout=7200 \
  `# 1h→2h: 큰 broadcast 테이블 처리` \
  \
  `# ===== Shuffle 최적화 (Multiple joins용) =====` \
  --conf spark.sql.shuffle.partitions=800 \
  `# 1000→800: 400 cores 대비 적절한 파티션 수` \
  --conf spark.default.parallelism=800 \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=800 \
  --conf spark.sql.adaptive.coalescePartitions.minPartitionSize=64MB \
  --conf spark.sql.adaptive.skewJoin.enabled=true \
  --conf spark.sql.adaptive.skewJoin.skewedPartitionFactor=5 \
  --conf spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes=256MB \
  `# Skew join: 사용자별 편향 처리` \
  \
  `# ===== 메모리 최적화 (캐싱 + XGBoost) =====` \
  --conf spark.memory.fraction=0.8 \
  `# 0.9→0.8: XGBoost off-heap 메모리 고려` \
  --conf spark.memory.storageFraction=0.5 \
  `# 0.4→0.5: 대량 캐싱을 위한 storage 메모리 증가` \
  --conf spark.executor.memoryOverhead=7g \
  `# 6g→7g: container 안정성 확보 (max 44GB 제한 고려)` \
  --conf spark.driver.maxResultSize=30g \
  `# 20g→30g: 모델 결과 수집` \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=1g \
  `# Off-heap: 1g로 감소 (YARN 44GB 제한 준수)` \
  \
  `# ===== Serialization (Kryo for XGBoost) =====` \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryo.registrationRequired=false \
  --conf spark.kryoserializer.buffer.max=2047m \
  `# 1g→2g: 큰 모델 객체 전송` \
  --conf spark.kryoserializer.buffer=512k \
  `# 256k→512k: 초기 버퍼 증가` \
  \
  `# ===== 압축 (네트워크 최적화) =====` \
  --conf spark.rdd.compress=true \
  --conf spark.shuffle.compress=true \
  --conf spark.shuffle.spill.compress=true \
  --conf spark.io.compression.codec=snappy \
  `# snappy: 속도 우선 (lz4도 가능)` \
  --conf spark.broadcast.compress=true \
  --conf spark.broadcast.blockSize=64m \
  \
  `# ===== Network & Timeout =====` \
  --conf spark.network.timeout=1800s \
  `# 1200s→1800s(30분): 노드 불안정 시 타임아웃 여유` \
  --conf spark.executor.heartbeatInterval=120s \
  `# 60s→120s: 학습 중 불필요한 heartbeat 감소` \
  --conf spark.rpc.message.maxSize=512 \
  `# 128→512MB: 큰 모델 전송` \
  --conf spark.hadoop.yarn.timeline-service.enabled=false \
  `# Timeline Server 연결 문제 회피` \
  --conf spark.excludeOnFailure.enabled=true \
  `# Bad node 자동 제외 (Spark 3.1.0+ 새 이름)` \
  --conf spark.excludeOnFailure.timeout=1h \
  --conf spark.excludeOnFailure.task.maxTaskAttemptsPerNode=2 \
  --conf spark.excludeOnFailure.stage.maxFailedTasksPerExecutor=2 \
  \
  `# ===== Dynamic Allocation =====` \
  --conf spark.dynamicAllocation.enabled=false \
  `# XGBoost barrier mode와 충돌하므로 비활성화 필수!` \
  --conf spark.shuffle.service.enabled=true \
  \
  `# ===== Locality (네트워크 빠른 환경) =====` \
  --conf spark.locality.wait=0 \
  `# Data locality보다 처리 속도 우선` \
  \
  `# ===== SQL 최적화 =====` \
  --conf spark.sql.sources.partitionOverwriteMode=dynamic \
  --conf spark.sql.optimizer.dynamicPartitionPruning.enabled=true \
  --conf spark.sql.autoBroadcastJoinThreshold=209715200 \
  --conf spark.sql.hive.filesourcePartitionFileCacheSize=1073741824 \
  `# (기본 256MB 수준) 파티션 메타 캐시 eviction 경고 방지` \
  --conf spark.sql.files.maxPartitionBytes=256MB \
  `# 128MB→256MB: Parquet 읽기 파티션 크기` \
  \
  `# ===== Task 설정 =====` \
  --conf spark.task.cpus=1 \
  --conf spark.task.maxFailures=4 \
  `# XGBoost 학습 중 일부 task 실패 허용` \
  \
  `# ===== XGBoost 관련 =====` \
  --conf spark.rapids.sql.enabled=false \
  `# GPU 미사용 (CPU 기반 XGBoost)` \
  \
  `# ===== JVM 최적화 =====` \
  --conf spark.driver.extraJavaOptions="-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=12 -XX:ParallelGCThreads=20 -XX:+ParallelRefProcEnabled -XX:MaxGCPauseMillis=200" \
  --conf spark.executor.extraJavaOptions="-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=2 -XX:ParallelGCThreads=5 -XX:+ParallelRefProcEnabled -XX:MaxGCPauseMillis=200" \
  `# Executor GC threads: Conc <= Parallel (container JVM crash 방지)` \
  \
  --jars /home/skinet/myfiles/data_bus/xgboost4j_2.12-3.1.1.jar,/home/skinet/myfiles/data_bus/xgboost4j-spark_2.12-3.1.1.jar,/home/skinet/myfiles/data_bus/synapseml_2.12-1.1.0.jar,/home/skinet/myfiles/data_bus/synapseml-core_2.12-1.1.0.jar,/home/skinet/myfiles/data_bus/synapseml-lightgbm_2.12-1.1.0.jar,/home/skinet/myfiles/data_bus/lightgbmlib-3.3.510.jar,/home/skinet/myfiles/data_bus/spray-json_2.12-1.3.6.jar,/home/skinet/myfiles/data_bus/spark-ml-feature-importance-helper-1.0.1.jar,/home/skinet/myfiles/data_driven_marketing/target/data_driven_marketing-1.1-SNAPSHOT.jar,/home/skinet/myfiles/data_bus/basicdataset-0.26.0.jar,/home/skinet/myfiles/data_bus/spark-excel_2.12-3.1.3_0.20.4.jar,/home/skinet/myfiles/data_bus/lightgbm-0.36.0.jar,/home/skinet/myfiles/data_bus/onnxruntime-engine-0.26.0.jar,/home/skinet/myfiles/data_bus/pytorch-engine-0.26.0.jar,/home/skinet/myfiles/data_bus/pytorch-model-zoo-0.26.0.jar,/home/skinet/myfiles/data_bus/pytorch-native-cpu-precxx11-2.1.1.jar,/home/skinet/myfiles/data_bus/pytorch-jni-2.1.1-0.26.0.jar,/home/skinet/myfiles/data_bus/sentencepiece-0.26.0.jar,/home/skinet/myfiles/data_bus/timeseries-0.26.0.jar,/home/skinet/myfiles/data_bus/tokenizers-0.26.0.jar,/home/skinet/myfiles/data_bus/xgboost-0.26.0.jar,/home/skinet/myfiles/data_bus/xgboost-gpu-0.26.0.jar,/home/skinet/myfiles/data_bus/spark-nlp-assembly-5.5.1.jar,/home/skinet/myfiles/data_bus/jsl-openvino-cpu_2.12-0.1.0.jar,/home/skinet/myfiles/data_bus/jsl-llamacpp-cpu_2.12-0.1.4.jar,/home/skinet/myfiles/data_bus/neo4j-connector-apache-spark_2.12-4.1.5_for_spark_3.jar,/home/skinet/myfiles/data_bus/graphframes-0.8.2-spark3.1-s_2.12.jar,/home/skinet/myfiles/data_bus/jmetalsp-spark-2.1-SNAPSHOT-jar-with-dependencies.jar,/home/skinet/myfiles/data_bus/jmetalsp-examples-2.1-SNAPSHOT-jar-with-dependencies.jar,/home/skinet/myfiles/data_bus/jmetalsp-spark-example-2.1-SNAPSHOT-jar-with-dependencies.jar,/home/skinet/myfiles/data_bus/ortools-linux-x86-64-9.8.3296.jar,/home/skinet/myfiles/data_bus/ortools-java-9.8.3296.jar,/home/skinet/myfiles/data_bus/jna-5.13.0.jar,/home/skinet/myfiles/data_bus/jna-platform-5.13.0.jar

# ===== 추가 최적화 옵션 (필요시 주석 해제) =====
#
# 1. 더 많은 메모리가 필요한 경우:
#    --conf spark.executor.memory=40g
#    --conf spark.executor.memoryOverhead=12g
#
# 2. Shuffle 성능 개선:
#    --conf spark.shuffle.file.buffer=1m
#    --conf spark.reducer.maxSizeInFlight=96m
#    --conf spark.shuffle.io.maxRetries=5
#
# 3. Speculation (느린 task 재실행):
#    --conf spark.speculation=true
#    --conf spark.speculation.multiplier=2
#
# 4. 데이터 편향이 심한 경우:
#    --conf spark.sql.adaptive.skewJoin.skewedPartitionFactor=10
#
# 5. Checkpoint (긴 lineage 대비):
#    spark.sparkContext.setCheckpointDir("hdfs://path/to/checkpoint")
#
# ===== 실행 가이드 =====
#
# 1. 개발/테스트 단계:
#    NUM_EXECUTORS=20
#    EXECUTOR_MEMORY="24g"
#    spark.sql.shuffle.partitions=200
#
# 2. 운영 단계:
#    현재 설정 사용 (80 executors)
#
# 3. 대규모 배치:
#    NUM_EXECUTORS=100
#    EXECUTOR_MEMORY="32g"
#    spark.sql.shuffle.partitions=1000
