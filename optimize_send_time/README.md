# Optimize Send Time - Work Guide

## 📂 Directory Overview

This directory contains Spark/Scala-based machine learning models for optimizing message send time prediction.

**File Structure:**
- `*.zpln` - Zeppelin notebook files (original source)
- `*.scala` - Pure Scala code files (converted from notebooks)
- `*.java` - Java implementation (GreedyAllocator for large-scale processing)

## 🎯 Project Purpose

- **Goal**: Predict optimal send times for marketing messages using XGBoost regression
- **Tech Stack**: Apache Spark 3.1.3, Scala 2.12.18, XGBoost
- **Data Source**: Campaign reaction data (MMS/RCS channels)

## 🚀 Quick Start

### 1. Environment Setup (First Time Only)

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/optimize_send_time

# Spark 환경 설치
./setup_spark_env.sh
source ~/.zshrc

# 최적화 라이브러리 다운로드
# jMetal 다운로드 (다목적 최적화용 - NEW!)
./download_jmetal.sh

# OR-Tools는 수동으로 다운로드하여 lib/ 디렉토리에 복사
# wget https://repo1.maven.org/maven2/com/google/ortools/ortools-java/9.4.1874/ortools-java-9.4.1874.jar -P lib/

# 환경 변수 설정 (권장)
source setup_all_optimizers.sh
# 이제 $JMETAL_JARS, $ORTOOLS_JARS, $ALL_OPTIMIZER_JARS 사용 가능
```

### 2. Generate Sample Data

```bash
# Option 1: Generate full dataset (100K users, 1M records)
./generate_sample_data.sh
# Select option 1

# Option 2: Generate small test dataset (1K users, 10K records)
./generate_sample_data.sh
# Select option 2
```

### 3. Run Quick Test

```bash
# Complete test with sample data
./quick_test_with_sample_data.sh
```

### 4. Interactive Development (권장 ⭐)

```bash
# 방법 1: Interactive Shell 시작
./run_interactive.sh

# Spark Shell 내부에서:
scala> :load optimize_ost.scala
scala> import OptimizeSendTime._
scala> val dfAll = spark.read.parquet("aos/sto/propensityScoreDF").cache()
```

> **중요**: spark-shell을 `-i optimize_ost.scala`로 시작한 경우, 같은 세션에서 다시 `:load optimize_ost.scala`를 실행하지 마세요.  
> 동일 정의 재로딩으로 인해 spark-shell이 크래시할 수 있습니다. (재로딩이 필요하면 `:quit` 후 재시작이 가장 안전합니다.)

```bash
# 방법 2: 자동 테스트
spark-shell -i load_and_test.scala

# 방법 3: Quick Run (대화형 메뉴)
./quick_run.sh
```

## 📊 Sample Data Schema

Generated sample data follows this schema:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `svc_mgmt_num` | String | User ID | s:0063c2994b5452d... |
| `send_ym` | String | Send year-month | 202512 |
| `send_hour` | Int | Send hour (9-18) | 10 |
| `propensity_score` | Double | Response probability | 0.7234 |

**Dataset Characteristics:**
- 100,000 users
- Each user has 10 records (one per hour: 9-18)
- Total: 1,000,000 records
- Propensity scores: 0.1 ~ 0.99
- Realistic distribution (users have preferred hours)

## 📋 AI Assistant Work Rules

### 1. Zeppelin Notebook (.zpln) Handling

#### Default Behavior
When working with `.zpln` files:
- **DO NOT** directly edit notebook files manually
- **ALWAYS** use Python scripts for automated conversion
- Convert to `.scala` files for version control and code review
- Maintain paragraph IDs in converted files

#### Conversion Process
```bash
python3 -c "
import json

with open('notebook.zpln', 'r') as f:
    notebook = json.load(f)

scala_code = []
for para in notebook['paragraphs']:
    text = para.get('text', '')
    para_id = para.get('id', '')
    
    # Skip PySpark paragraphs
    if text.strip().startswith('%pyspark'):
        continue
    
    # Remove Zeppelin directives
    if text.strip().startswith('%'):
        text = '\\n'.join(text.split('\\n')[1:])
    
    # Remove Zeppelin-specific code
    if 'z.show(' in text or 'z.input(' in text:
        continue
        
    if text.strip():
        scala_code.append(f'// ===== {para_id} =====')
        scala_code.append(text)
        scala_code.append('')

with open('output.scala', 'w') as f:
    f.write('\\n'.join(scala_code))
"
```

### 2. Zeppelin-Specific Code Removal

When converting `.zpln` to `.scala`, **ALWAYS** remove or replace:

#### Must Remove:
- `z.show()` - Zeppelin display function
- `z.run()` - Zeppelin paragraph execution
- `z.angular()` - Zeppelin angular binding
- `%sql`, `%md`, `%pyspark` - Interpreter directives

#### Must Replace:
- `z.input("name", "default")` → `val name = "default"` (with comment explaining source)

#### Example Replacement:
```scala
// Original (Zeppelin):
val smnSuffix = z.input("suffix", "0").toString
z.show(dataDF)

// Converted (Pure Scala):
// Default suffix value (previously from Zeppelin input widget)
val smnSuffix = "0"
// To view data, use: dataDF.show()
```

### 3. Paragraph ID Preservation

**ALWAYS** preserve original paragraph IDs when converting:

```scala
// ===== paragraph_1764658338256_686533166 =====
import com.microsoft.azure.synapse.ml.causal
// ... code ...

// ===== paragraph_1764742922351_426209997 =====
def getPreviousMonths(startMonthStr: String, periodM: Int): Array[String] = {
// ... code ...
}
```

**Why?**
- Easy mapping between notebook and Scala files
- Better debugging and issue tracking
- Clear code organization by logical sections

### 4. Spark Environment Setup

#### Local Spark Installation
Location: `~/spark-local/`
- Spark 3.1.3
- Scala 2.12.18
- Java 8 (OpenJDK 1.8.0_292)

#### Environment Activation
```bash
source ~/spark-local/spark-env.sh
```

#### Running Scala Code
```bash
# Interactive Shell
spark-shell

# Execute Scala file
spark-shell -i /path/to/file.scala

# With specific memory settings
spark-shell --driver-memory 4g --executor-memory 4g
```

### 5. XGBoost Model Configuration

When modifying XGBoost models:

#### Standard Parameters
```scala
val xgbParamR = Map(
  "eta" -> 0.01,                    // Learning rate
  "max_depth" -> 6,                 // Tree depth
  "objective" -> "reg:squarederror", // Regression task
  "num_round" -> 100,               // Number of iterations
  "num_workers" -> 10,              // Parallel workers
  "eval_metric" -> "rmse"           // Evaluation metric
)
```

#### Feature Interaction Constraints
For forcing specific features (e.g., `send_hournum_cd`) as primary split:

```scala
// Find feature indices
val assemblerInputCols = vectorAssembler.getInputCols
val sendHournumIdx = assemblerInputCols.indexOf("send_hournum_cd_enc")

// Apply constraints
val xgbParamWithConstraints = xgbParamR + (
  "interaction_constraints" -> s"[[$sendHournumIdx],[0,1,2,...]]"
)
```

### 6. Code Style Guidelines

#### Imports Organization
```scala
// 1. External libraries
import com.microsoft.azure.synapse.ml.causal
import ml.dmlc.xgboost4j.scala.spark._

// 2. Spark libraries
import org.apache.spark.ml.classification._
import org.apache.spark.sql.functions.{col, lit, expr}

// 3. Java standard libraries
import java.time.format.DateTimeFormatter

// 4. Scala standard libraries
import scala.collection.mutable.ListBuffer
```

#### Variable Naming
- Use `camelCase` for variables: `sendMonth`, `featureYmList`
- Use `PascalCase` for classes/objects: `XGBoostRegressor`
- Use descriptive names: `resDFFiltered` not `df1`

#### SQL Queries
```scala
// Prefer multiline formatting for readability
val df = spark.sql("""
  SELECT 
    svc_mgmt_num,
    send_dt,
    click_yn
  FROM tos.od_tcam_cmpgn_obj_cont
  WHERE send_ym = '202512'
""")
```

### 7. Git Workflow

#### Before Committing
1. Convert `.zpln` files to `.scala`
2. Remove all Zeppelin-specific code
3. Verify Scala syntax (no compilation errors)
4. Test key functionality if possible

#### Commit Message Format
```bash
git commit -m "feat: Add XGBoost model with feature constraints

- Implement interaction constraints for send_hournum_cd
- Update parameter tuning for better RMSE
- Files: predict_ost_251221.scala
"
```

#### Files to Track
- ✅ `.scala` files (always)
- ✅ `.zpln` files (for reference)
- ❌ `.metals/`, `.scala-build/` (ignore)

### 8. Testing & Validation

#### Quick Syntax Check
```bash
# Try to load in Spark shell
spark-shell -i predict_ost_251221.scala
```

#### Common Issues
1. **Missing variable**: Check if Zeppelin `z.input()` was removed
   - Fix: Add default value declaration
   
2. **Import errors**: Verify all imports are valid for local Spark
   - Fix: Remove Synapse-specific imports if needed

3. **SQL syntax errors**: Check string interpolation
   - Fix: Use proper `s"..."` or `f"..."` syntax

## 🧪 Usage Examples

### Example 1: Greedy Allocation (Fastest)

```scala
import OptimizeSendTime._

val dfAll = spark.read.parquet("data/sample/propensityScoreDF").cache()
val df = dfAll.limit(10000)

val capacity = Map(
  9 -> 1000, 10 -> 1000, 11 -> 1000, 12 -> 1000, 13 -> 1000,
  14 -> 1000, 15 -> 1000, 16 -> 1000, 17 -> 1000, 18 -> 1000
)

val result = allocateGreedySimple(df, Array(9,10,11,12,13,14,15,16,17,18), capacity)
result.show()
```

### Example 2: Simulated Annealing (Balanced)

```scala
import OptimizeSendTime._

val dfAll = spark.read.parquet("data/sample/propensityScoreDF").cache()
val df = dfAll.limit(10000)

val capacityMap = Map(
  9 -> 1000, 10 -> 1000, 11 -> 1000, 12 -> 1000, 13 -> 1000,
  14 -> 1000, 15 -> 1000, 16 -> 1000, 17 -> 1000, 18 -> 1000
)

val result = allocateUsersWithSimulatedAnnealing(
  df = df,
  capacityPerHour = capacityMap,
  maxIterations = 100000,
  initialTemperature = 1000.0,
  coolingRate = 0.9995,
  batchSize = 500000
)

result.groupBy("assigned_hour").count().orderBy("assigned_hour").show()
```

### Example 3: OR-Tools Optimization (Most Accurate)

```scala
import OptimizeSendTime._

val dfAll = spark.read.parquet("data/sample/propensityScoreDF").cache()
val df = dfAll.limit(5000)  // OR-Tools works best with smaller datasets

val capacityMap = Map(
  9 -> 500, 10 -> 500, 11 -> 500, 12 -> 500, 13 -> 500,
  14 -> 500, 15 -> 500, 16 -> 500, 17 -> 500, 18 -> 500
)

val result = allocateUsersWithHourlyCapacity(
  df = df,
  capacityPerHour = capacityMap,
  timeLimit = 300,
  topChoices = 5,
  enablePreprocessing = true
)

result.show()
```

### Example 3-1: jMetal NSGA-II (Multi-Objective - NEW!)

```bash
# 환경 변수 설정 (처음 한 번)
source setup_all_optimizers.sh

# Spark shell 시작
spark-shell --jars $JMETAL_JARS -i optimize_ost.scala
```

```scala
import OptimizeSendTime._

val dfAll = spark.read.parquet("data/sample/propensityScoreDF").cache()
val df = dfAll.limit(10000)

val capacityMap = Map(
  9 -> 1000, 10 -> 1000, 11 -> 1000, 12 -> 1000, 13 -> 1000,
  14 -> 1000, 15 -> 1000, 16 -> 1000, 17 -> 1000, 18 -> 1000
)

// 다목적 최적화: 점수 최대화 + 부하 분산
val result = allocateUsersWithJMetalNSGAII(
  df = df,
  capacityPerHour = capacityMap,
  populationSize = 100,
  maxEvaluations = 25000
)

result.groupBy("assigned_hour").count().orderBy("assigned_hour").show()
```

### Example 4: Large Scale Processing

```scala
import OptimizeSendTime._

val dfAll = spark.read.parquet("data/sample/propensityScoreDF").cache()

val capacityMap = Map(
  9 -> 10000, 10 -> 10000, 11 -> 10000, 12 -> 10000, 13 -> 10000,
  14 -> 10000, 15 -> 10000, 16 -> 10000, 17 -> 10000, 18 -> 10000
)

val result = allocateLargeScaleHybrid(
  df = dfAll,
  capacityPerHour = capacityMap,
  batchSize = 50000,
  timeLimit = 300,
  topChoices = 5,
  enablePreprocessing = true
)

// Save results
result.write.mode("overwrite").parquet("data/results/allocation_result")
```

## 📚 Common Operations

### Generate Sample Data
```bash
# Full dataset (100K users)
./generate_sample_data.sh

# Or in Spark Shell
import GenerateSampleData._
val df = generateSampleData(spark, numUsers = 100000, sendYm = "202512")
```

### Convert New Zeppelin Notebook
```bash
# 1. Place .zpln file in optimize_send_time/
# 2. Run conversion script (see section 1)
# 3. Remove Zeppelin-specific code
# 4. Add default values for z.input() parameters
# 5. Verify paragraph IDs are preserved
```

### Update Existing Model
```bash
# 1. Edit .scala file directly
# 2. Test changes in spark-shell
# 3. Update corresponding .zpln if needed
# 4. Commit both files
```

### Add New Features
```bash
# 1. Add feature engineering code
# 2. Update VectorAssembler input columns
# 3. Adjust XGBoost parameters if needed
# 4. Document changes in comments
```

## 🔍 Key Files

### Optimization System (New)
| File | Purpose | Status |
|------|---------|--------|
| `optimize_ost.scala` | User allocation optimizer (OR-Tools, SA, jMetal) | **Production** |
| `generate_sample_data.scala` | Sample data generator | Development |
| `setup_spark_env.sh` | Environment setup script | Setup |
| `setup_ortools_jars.sh` | OR-Tools JAR downloader | Setup |
| `download_jmetal.sh` | ⭐ jMetal JAR downloader (NEW) | **Setup** |
| `setup_jmetal_env.sh` | jMetal 환경 변수 설정 | Setup |
| `setup_all_optimizers.sh` | ⭐ 모든 라이브러리 환경 변수 통합 (NEW) | **Recommended** |
| `spark-shell-jmetal.sh` | jMetal 지원 Spark shell 래퍼 | Convenience |
| `example_jmetal.scala` | jMetal 사용 예제 | Example |
| `run_optimize_ost.sh` | Execution script | Development |
| `generate_sample_data.sh` | Data generation script | Development |
| `generate_data_simple.sh` | Simple data generator | Development |
| `generate_data_with_backup.sh` | Data generator with backup | Development |
| `quick_test_with_sample_data.sh` | Quick test script | Testing |
| `quick_run.sh` | Quick run with auto-detection | Development |
| `run_interactive.sh` | ⭐ Interactive shell launcher | **Recommended** |
| `load_and_test.scala` | Auto-load and test script | Testing |

### Prediction Models (Original)
| File | Purpose | Status |
|------|---------|--------|
| `predict_ost_251221.zpln` | Latest notebook (Dec 2024) | Source |
| `predict_ost_251221.scala` | Converted Scala code | Archive |
| `predict_ost_25121711.zpln` | Previous version | Archive |
| `predict_ost_25121711.scala` | Previous Scala code | Archive |
| `predict_ost_25121710.scala` | Oldest version | Archive |

### Documentation
| File | Purpose |
|------|---------|
| `INSTALLATION_GUIDE.md` | Detailed installation guide (macOS) |
| `LINUX_DEPLOYMENT_GUIDE.md` | Linux server deployment guide |
| `DATA_GENERATION_GUIDE.md` | Sample data generation guide |
| `TROUBLESHOOTING.md` | ⭐ Problem solving guide |
| `QUICK_START.md` | 5-minute quick start guide |
| `JMETAL_SETUP.md` | ⭐ jMetal 설정 및 사용 가이드 (NEW) |
| `QUICK_START_JMETAL.md` | jMetal 빠른 시작 (NEW) |
| `ENV_SETUP_GUIDE.md` | ⭐ 환경 변수 설정 가이드 (NEW) |
| `README_SBT.md` | SBT project guide (after setup) |

## 🎯 알고리즘 비교 (NEW!)

| 알고리즘 | 목적 | 속도 | 품질 | 메모리 | 권장 사용 |
|---------|------|------|------|--------|----------|
| **Greedy** | 빠른 할당 | ⚡⚡⚡⚡⚡ | ⭐⭐ | 💾 | 초기 테스트, 대용량 |
| **Simulated Annealing** | 준최적해 탐색 | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 💾💾 | 밸런스형, 중대규모 |
| **OR-Tools** | 정확한 최적해 | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💾💾 | 소규모, 최고 품질 |
| **jMetal NSGA-II** | 다목적 최적화 | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💾💾💾 | 부하분산+점수 |
| **jMetal MOEA/D** | 빠른 다목적 | ⚡⚡⚡⚡ | ⭐⭐⭐ | 💾💾 | 빠른 수렴 필요시 |

### 알고리즘 선택 가이드

```bash
# 상황 1: 100만+ 사용자, 빠른 결과 필요
→ Greedy 또는 Hybrid (Greedy 기반)

# 상황 2: 10만 사용자, 높은 품질 필요
→ Simulated Annealing 또는 Hybrid (SA 기반)

# 상황 3: 5만 이하 사용자, 최고 품질 필요
→ OR-Tools 또는 Hybrid (OR-Tools 기반)

# 상황 4: 점수와 부하분산 모두 중요
→ jMetal NSGA-II 또는 MOEA/D

# 상황 5: 대용량 + 높은 품질
→ allocateLargeScaleHybrid 또는 allocateLargeScaleJMetal
```

## 💡 Best Practices

### For Optimization System
1. **Start small** - Test with limited data first (limit 1000-10000)
2. **Use Greedy for quick tests** - Fastest way to validate data and logic
3. **Monitor memory** - Watch Spark UI at http://localhost:4040
4. **Adjust batch size** - Reduce if you encounter OOM errors
5. **Use preprocessing** - Enable `topChoices` to reduce problem size
6. **Save intermediate results** - Cache DataFrames for reuse

### For Zeppelin Notebooks
1. **Always preserve paragraph IDs** - Enables mapping between notebooks and code
2. **Remove Zeppelin dependencies** - Ensures code runs in pure Spark environment
3. **Add explanatory comments** - Especially for replaced Zeppelin widgets
4. **Test before committing** - At minimum, verify Scala syntax
5. **Use descriptive variable names** - Makes code maintainable
6. **Format SQL queries** - Multi-line format for readability
7. **Document model parameters** - Explain non-obvious hyperparameter choices

## 🚨 Common Pitfalls

### Optimization System
1. ❌ **Out of Memory** → Reduce batch size or limit dataset
2. ❌ **Slow execution** → Use preprocessing, reduce iterations, or use Greedy
3. ❌ **Wrong data path** → Verify parquet file location
4. ❌ **Java not found** → Run `source ~/.zshrc` after setup
5. ❌ **No results** → Check capacity vs user count ratio

### Zeppelin Notebooks
1. ❌ Leaving `z.show()` in code → Replace with `.show()` or remove
2. ❌ Undefined variables from removed `z.input()` → Add default declarations
3. ❌ Wrong Scala version → Use Scala 2.12.18
4. ❌ Missing environment activation → Source `spark-env.sh` first
5. ❌ Lost paragraph IDs → Always extract from original .zpln

## 📞 Support

For issues specific to this directory:
1. Check paragraph ID in .scala file
2. Find corresponding paragraph in .zpln file
3. Review original Zeppelin code
4. Verify environment setup (Spark/Scala versions)

## 🎓 Learning Path

1. ✅ **Setup** - Run `./setup_spark_env.sh`
2. ✅ **Generate Data** - Run `./generate_sample_data.sh`
3. ✅ **Quick Test** - Run `./quick_test_with_sample_data.sh`
4. ✅ **Try Greedy** - Test with 10K records
5. ✅ **Try SA** - Test with 1K records, 10K iterations
6. ✅ **Parameter Tuning** - Adjust iterations, temperature, batch size
7. ✅ **Scale Up** - Test with full dataset
8. ✅ **Production** - Deploy to Red Hat server

## 📖 Additional Resources

- **Detailed Setup**: See `INSTALLATION_GUIDE.md`
- **Quick Examples**: See `QUICK_START.md`
- **SBT Build**: See `README_SBT.md` (after running `create_sbt_project.sh`)
- **Spark UI**: http://localhost:4040 (during execution)
- **Spark Docs**: https://spark.apache.org/docs/3.1.3/

---

## ☕ Java 버전 (NEW)

Scala의 `greedy_allocation.scala`를 Java로 완전 변환한 구현이 제공됩니다.

### 특징
- ✅ Java 8+ 호환
- ✅ Spark Java API 사용
- ✅ 동일한 성능 (~1-5% 차이)
- ✅ Maven/Gradle 빌드 지원

### 빌드 및 실행
```bash
# 빌드
./build_java.sh

# 실행
spark-submit \
  --class optimize_send_time.GreedyAllocatorTest \
  --driver-memory 16g \
  --executor-memory 16g \
  build/greedy-allocator.jar
```

### 사용 예제
```java
// 1. Allocator 생성
GreedyAllocator allocator = new GreedyAllocator();

// 2. 데이터 로드
Dataset<Row> df = spark.read().parquet("aos/sto/propensityScoreDF");

// 3. 용량 설정
int[] hours = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
Map<Integer, Integer> capacity = new HashMap<>();
for (int h : hours) capacity.put(h, 2500000);

// 4. 할당 실행
Dataset<Row> result = allocator.allocateLargeScale(
    df, hours, capacity, 1000000
);
```

**자세한 가이드**: `JAVA_USAGE.md`

---

**Last Updated**: 2026-01-14
**Maintainer**: Data Science Team

