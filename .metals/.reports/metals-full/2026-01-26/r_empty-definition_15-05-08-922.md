error id: file://<WORKSPACE>/predict_send_time/data_transformation.scala:
file://<WORKSPACE>/predict_send_time/data_transformation.scala
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:

offset: 7420
uri: file://<WORKSPACE>/predict_send_time/data_transformation.scala
text:
```scala
// =============================================================================
// MMS Click Prediction - Data Transformation
// =============================================================================
// 이 코드는 raw_data_generation.scala에서 생성된 통합 raw 데이터를 읽어서:
// 1. Train/Test split 수행
// 2. Feature engineering pipeline 적용 (transformation)
// 3. Transformed data 저장
//
// [실행 방법]
// 1. Paragraph 1-7: 순차 실행 (전체 데이터 처리)
// =============================================================================


// ===== Paragraph 1: Imports and Configuration =====

import com.microsoft.azure.synapse.ml.causal
import com.skt.mno.dt.utils.commfunc._
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.types.{DateType, StringType, TimestampType}
import org.apache.spark.sql.{DataFrame, SparkSession, functions => F}
import org.apache.spark.storage.StorageLevel
import org.joda.time.{LocalDate, Years}

import java.sql.Date
import java.text.DecimalFormat
import java.time.format.DateTimeFormatter
import java.time.{ZoneId, ZonedDateTime}
import com.microsoft.azure.synapse.{ml => sml}

import collection.JavaConverters._
import scala.collection.JavaConverters._

// 대용량 처리 최적화를 위한 Spark 설정
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "50m")

// 메모리 최적화 설정 (OOM 방지)
spark.conf.set("spark.executor.memoryOverhead", "4g")
spark.conf.set("spark.memory.fraction", "0.8")
spark.conf.set("spark.memory.storageFraction", "0.3")

println("Spark configuration set for data transformation")


// ===== Paragraph 2: Helper Functions =====

import java.time.{YearMonth, LocalDate}
import java.time.format.DateTimeFormatter

// 이전 N개월의 월 리스트 생성 (역순: 최근 → 과거)
def getPreviousMonths(startMonthStr: String, periodM: Int): Array[String] = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMM")
  val startMonth = YearMonth.parse(startMonthStr, formatter)
  var resultMonths = scala.collection.mutable.ListBuffer[String]()
  var currentMonth = startMonth

  for (i <- 0 until periodM) {
    resultMonths.prepend(currentMonth.format(formatter))
    currentMonth = currentMonth.minusMonths(1)
  }
  resultMonths.toArray
}

// 날짜 문자열(yyyyMMdd)에서 월 문자열(yyyyMM) 추출
def getMonthFromDate(dateStr: String): String = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
  val date = LocalDate.parse(dateStr, formatter)
  date.format(DateTimeFormatter.ofPattern("yyyyMM"))
}

// 날짜 문자열(yyyyMMdd)에서 하루 전 날짜의 월 문자열(yyyyMM) 추출
def getPreviousDayMonth(dateStr: String): String = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
  val date = LocalDate.parse(dateStr, formatter)
  val previousDay = date.minusDays(1)
  previousDay.format(DateTimeFormatter.ofPattern("yyyyMM"))
}

println("Helper functions defined")


// ===== Paragraph 3: Configuration Variables =====

// =============================================================================
// 시간 조건 및 경로 설정
// =============================================================================

// Raw data 버전 설정
val rawDataVersion = "1"  // raw_data_generation.scala에서 생성한 버전과 일치

// Train/Test split 기준 날짜 (먼저 선언)
val predictionDTSta = "20251201"  // 테스트 데이터 시작 날짜
val predictionDTEnd = "20260101"  // 테스트 데이터 종료 날짜

// 학습 기간 설정 (월 단위) - predictionDTSta 기반 자동 계산
val sendMonth = getPreviousDayMonth(predictionDTSta)  // predictionDTSta 하루 전의 월 (예: 20251201 → 202511)
val period = 6              // 학습 기간 (개월 수)
val sendYmList = getPreviousMonths(sendMonth, period)  // 학습용 월 리스트

// Transformed data 저장 버전
val transformedDataVersion = "1"  // 저장할 버전 번호

// Pipeline fitting 샘플링 비율 (메모리 절약)
val pipelineSampleRate = 0.3

// Suffix 배치 저장 설정 (Sliding Window)
val suffixGroupSize = 2  // 한 번에 처리할 suffix 개수
val suffixSlide = 2      // Slide 크기 (suffixGroupSize와 같으면 overlap 없음)

// 시간대 설정
val startHour = 9
val endHour = 18
val hourRange = (startHour to endHour).toList

println("=" * 80)
println("Configuration Summary")
println("=" * 80)
println(s"Raw Data Version: $rawDataVersion")
println(s"Transformed Data Version: $transformedDataVersion")
println(s"Test Period: $predictionDTSta ~ $predictionDTEnd")
println(s"Training Month (auto-calculated): $sendMonth (1 day before $predictionDTSta)")
println(s"Training Period: $period months (${sendYmList.mkString(", ")})")
println(s"Train/Test Split: send_ym in (${sendYmList.mkString(", ")}) (train) / send_dt >= $predictionDTSta (test)")
println(s"Pipeline Sample Rate: $pipelineSampleRate")
println(s"Suffix Batch Save (Sliding Window):")
println(s"  - Group size: $suffixGroupSize")
println(s"  - Slide: $suffixSlide")
println("=" * 80)


// ===== Paragraph 4: Raw Data Loading and Train/Test Split =====

// =============================================================================
// Raw Data 로딩 및 Train/Test 분리
// =============================================================================

println("=" * 80)
println("Loading raw data and splitting into train/test...")
println("=" * 80)

val rawDataPath = s"aos/sto/rawDF${rawDataVersion}"
println(s"Loading from: $rawDataPath")

val rawDF = spark.read.parquet(rawDataPath)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println(s"Raw data loaded and cached")

// Train/Test split
println(s"Splitting data:")
println(s"  - Training: send_ym in (${sendYmList.mkString(", ")})")
println(s"  - Testing:  send_dt >= $predictionDTSta and send_dt < $predictionDTEnd")

// Training 데이터: 학습 기간(월) 기반 필터링
val trainDFRev = rawDF
    .filter(s"""send_ym in (${sendYmList.mkString("'","','","'")})""")
    .withColumn("hour_gap", F.expr("case when res_utility>=1.0 then 1 else 0 end"))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

// Test 데이터: 날짜 기반 필터링 (테스트 기간)
val testDFRev = rawDF
    .filter(F.col("send_dt") >= predictionDTSta)
    .filter(F.col("send_dt") < predictionDTEnd)
    .withColumn("hour_gap", F.expr("case when res_utility>=1.0 then 1 else 0 end"))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println(s"Split completed:")
println(s"  - Training period: ${sendYmList.head} ~ ${sendYmList.last} ($period months)")
println(s"  - Test period: $predictionDTSta ~ $predictionDTEnd")
println("=" * 80)

// Raw data unpersist (메모리 확보)
rawDF.unpersist()


// ===== Paragraph 5: Feature Column Analysis =====

// =============================================================================
// 피처 컬럼 분석
// =============================================================================

println("=" * 80)
println("Analyzing feature columns...")
println("=" * 80)

val noFeatureCols = Array("click_yn", "hour_gap", "chnl_typ", "cmpgn_typ")
val tokenCols = trainDFRev.columns.filter(x => x.endsWith("_token")).distinct
val continuousCols = (trainDFRev.columns
    .filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false))
    .distinct
    .filter(x => !t@@okenCols.contains(x) && !noFeatureCols.contains(x))
).distinct
val categoryCols = (trainDFRev.columns
    .filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false))
    .distinct
    .filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))
).distinct
val vectorCols = trainDFRev.columns.filter(x => x.endsWith("_vec"))

println(s"Feature column summary:")
println(s"  - Token columns: ${tokenCols.length}")
tokenCols.sorted.foreach(c => println(s"    * $c"))
println(s"  - Continuous columns: ${continuousCols.length}")
continuousCols.sorted.foreach(c => println(s"    * $c"))
println(s"  - Category columns: ${categoryCols.length}")
categoryCols.sorted.foreach(c => println(s"    * $c"))
println(s"  - Vector columns: ${vectorCols.length}")
vectorCols.sorted.foreach(c => println(s"    * $c"))
println("=" * 80)


// ===== Paragraph 6: Pipeline Definition and Transformation =====

// =============================================================================
// Feature Engineering Pipeline 생성 및 적용
// =============================================================================

println("=" * 80)
println("Creating and fitting feature engineering pipelines...")
println("=" * 80)

// Pipeline 파라미터 설정
val tokenColsEmbCols = Array("app_usage_token")
val featureHasherNumFeature = 128

var nodeNumber = 10
var coreNumber = 32
try {
    nodeNumber = spark.conf.get("spark.executor.instances").toInt
    coreNumber = spark.conf.get("spark.executor.cores").toInt
} catch {
    case ex: Exception => {}
}

val params: Map[String, Any] = Map(
    "minDF" -> 1,
    "minTF" -> 5,
    "embSize" -> 30,
    "vocabSize" -> 1000,
    "numParts" -> nodeNumber
)

// Feature column 설정
val labelColClick = "click_yn"
val labelColGap = "hour_gap"

val indexedLabelColClick = "click_yn"
val indexedLabelColGap = "hour_gap"

val indexedFeatureColClick = "indexedFeaturesClick"
val scaledFeatureColClick = "scaledFeaturesClick"
val selectedFeatureColClick = "selectedFeaturesClick"

val indexedFeatureColGap = "indexedFeaturesGap"
val scaledFeatureColGap = "scaledFeaturesGap"
val selectedFeatureColGap = "selectedFeaturesGap"

val onlyGapFeature = Array[String]()

val doNotHashingCateCols = Array[String]("send_hournum_cd", "peak_usage_hour_cd")
val doNotHashingContCols = Array[String](
    "click_cnt",
    "heavy_usage_app_cnt", "medium_usage_app_cnt", "light_usage_app_cnt",
    "total_traffic_mb", "app_cnt", "peak_hour_app_cnt", "active_hour_cnt",
    "avg_hourly_app_avg", "total_daily_traffic_mb", "total_heavy_apps_cnt",
    "total_medium_apps_cnt", "total_light_apps_cnt"
)

// Pipeline 생성 함수
def makePipeline(
    labelCols: Array[Map[String, String]] = Array.empty,
    indexedFeatureCol: String = "indexed_features",
    scaledFeatureCol: String = "scaled_features",
    selectedFeatureCol: String = "selected_features",
    tokenCols: Array[String] = Array.empty,
    vectorCols: Array[String] = Array.empty,
    continuousCols: Array[String] = Array.empty,
    categoryCols: Array[String] = Array.empty,
    doNotHashingCateCols: Array[String] = Array.empty,
    doNotHashingContCols: Array[String] = Array.empty,
    useSelector: Boolean = false,
    useScaling: Boolean = false,
    tokenColsEmbCols: Array[String] = Array.empty,
    featureHasherNumFeature: Int = 256,
    featureHashColNm: String = "feature_hashed",
    colNmSuffix: String = "#",
    params: Map[String, Any],
    userDefinedFeatureListForAssembler: Array[String] = Array.empty
) = {
    val minDF = params.get("minDF").getOrElse(1).asInstanceOf[Int]
    val minTF = params.get("minTF").getOrElse(5).asInstanceOf[Int]
    val embSize = params.get("embSize").getOrElse(10).asInstanceOf[Int]
    val vocabSize = params.get("vocabSize").getOrElse(30).asInstanceOf[Int]
    val numParts = params.get("numParts").getOrElse(10).asInstanceOf[Int]

    var featureListForAssembler = continuousCols ++ vectorCols

    import org.apache.spark.ml.{Pipeline, PipelineStage}
    val transformPipeline = new Pipeline().setStages(Array[PipelineStage]())

    if (labelCols.size > 0) {
        labelCols.foreach(map =>
            if (transformPipeline.getStages.isEmpty) {
                transformPipeline.setStages(Array(new StringIndexer(map("indexer_nm")).setInputCol(map("col_nm")).setOutputCol(map("label_nm")).setHandleInvalid("skip")))
            } else {
                transformPipeline.setStages(transformPipeline.getStages ++ Array(new StringIndexer(map("indexer_nm")).setInputCol(map("col_nm")).setOutputCol(map("label_nm")).setHandleInvalid("skip")))
            }
        )
    }

    val tokenColsEmb = tokenCols.filter(x => tokenColsEmbCols.map(x.contains(_)).reduceOption(_ || _).getOrElse(false))
    val tokenColsCnt = tokenCols.filter(!tokenColsEmb.contains(_))

    if (embSize > 0 && tokenColsEmb.size > 0) {
        val embEncoder = tokenColsEmb.map(c => new Word2Vec().setNumPartitions(numParts).setSeed(46).setVectorSize(embSize).setMinCount(minTF).setInputCol(c).setOutputCol(c + "_embvec"))
        transformPipeline.setStages(if (transformPipeline.getStages.isEmpty) {
            embEncoder
        } else {
            transformPipeline.getStages ++ embEncoder
        })
        featureListForAssembler ++= tokenColsEmb.map(_ + "_" + colNmSuffix + "_embvec")
    }

    if (tokenColsCnt.size > 0) {
        // CountVectorizer + TF-IDF (정보 손실 최소화)
        val cntVectorizer = tokenColsCnt.map(c =>
            new CountVectorizer()
                .setInputCol(c)
                .setOutputCol(c + "_" + colNmSuffix + "_cntvec")
                .setVocabSize(1000) // 정확도 우선
                .setMinDF(3) // 노이즈 제거
                .setBinary(false) // 빈도 정보 유지
        )
        transformPipeline.setStages(if (transformPipeline.getStages.isEmpty) {
            cntVectorizer
        } else {
            transformPipeline.getStages ++ cntVectorizer
        })

        // TF-IDF 가중치 추가
        val tfidfTransformers = tokenColsCnt.map(c =>
            new IDF()
                .setInputCol(c + "_" + colNmSuffix + "_cntvec")
                .setOutputCol(c + "_" + colNmSuffix + "_tfidf")
        )
        transformPipeline.setStages(transformPipeline.getStages ++ tfidfTransformers)

        featureListForAssembler ++= tokenColsCnt.map(_ + "_" + colNmSuffix + "_tfidf")
    }

    if (featureHasherNumFeature > 0 && categoryCols.size > 0) {
        val featureHasher = new FeatureHasher().setNumFeatures(featureHasherNumFeature)
            .setInputCols((continuousCols ++ categoryCols)
                .filter(c => !doNotHashingContCols.contains(c))
                .filter(c => !doNotHashingCateCols.contains(c))
            ).setOutputCol(featureHashColNm)

        transformPipeline.setStages(if (transformPipeline.getStages.isEmpty) {
            Array(featureHasher)
        } else {
            transformPipeline.getStages ++ Array(featureHasher)
        })
        featureListForAssembler = featureListForAssembler.filter(!continuousCols.contains(_))
        featureListForAssembler ++= Array(featureHashColNm)
    }

    if (doNotHashingCateCols.size > 0) {
        val catetoryIndexerList = doNotHashingCateCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_" + colNmSuffix + "_index").setHandleInvalid("keep"))
        val encoder = new OneHotEncoder().setInputCols(doNotHashingCateCols.map(c => c + "_" + colNmSuffix + "_index")).setOutputCols(doNotHashingCateCols.map(c => c + "_" + colNmSuffix + "_enc")).setHandleInvalid("keep")
        transformPipeline.setStages(if (transformPipeline.getStages.isEmpty) {
            catetoryIndexerList ++ Array(encoder)
        } else {
            transformPipeline.getStages ++ catetoryIndexerList ++ Array(encoder)
        })
        featureListForAssembler ++= doNotHashingCateCols.map(_ + "_" + colNmSuffix + "_enc")
    }

    if (doNotHashingContCols.size > 0) {
        featureListForAssembler ++= doNotHashingContCols
    }

    if (featureHasherNumFeature < 1 && categoryCols.size > 0) {
        val catetoryIndexerList = categoryCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_" + colNmSuffix + "_index").setHandleInvalid("keep"))
        val encoder = new OneHotEncoder().setInputCols(categoryCols.map(c => c + "_" + colNmSuffix + "_index")).setOutputCols(categoryCols.map(c => c + "_" + colNmSuffix + "_enc")).setHandleInvalid("keep")
        transformPipeline.setStages(if (transformPipeline.getStages.isEmpty) {
            catetoryIndexerList ++ Array(encoder)
        } else {
            transformPipeline.getStages ++ catetoryIndexerList ++ Array(encoder)
        })
        featureListForAssembler ++= categoryCols.map(_ + "_" + colNmSuffix + "_enc")
    }

    if (userDefinedFeatureListForAssembler.size > 0) {
        featureListForAssembler ++= userDefinedFeatureListForAssembler
    }

    val assembler = new VectorAssembler().setInputCols(featureListForAssembler.distinct).setOutputCol(indexedFeatureCol).setHandleInvalid("keep")

    transformPipeline.setStages(transformPipeline.getStages ++ Array(assembler))

    if (useSelector) {
        val selector = new VarianceThresholdSelector().setVarianceThreshold(8.0).setFeaturesCol(indexedFeatureCol).setOutputCol(selectedFeatureCol)
        transformPipeline.setStages(transformPipeline.getStages ++ Array(selector))
    }

    if (useScaling) {
        val inputFeautreCol = if (useSelector) {
            selectedFeatureCol
        } else {
            indexedFeatureCol
        }
        val scaler = new MinMaxScaler().setInputCol(inputFeautreCol).setOutputCol(scaledFeatureCol)
        transformPipeline.setStages(transformPipeline.getStages ++ Array(scaler))
    }

    transformPipeline
}

// Click 모델용 Pipeline
println("Creating Click transformer pipeline...")
val transformPipelineClick = makePipeline(
    labelCols = Array(),
    indexedFeatureCol = indexedFeatureColClick,
    scaledFeatureCol = scaledFeatureColClick,
    selectedFeatureCol = selectedFeatureColClick,
    tokenCols = Array("app_usage_token"),
    vectorCols = vectorCols.filter(!onlyGapFeature.contains(_)),
    continuousCols = continuousCols.filter(!onlyGapFeature.contains(_)),
    categoryCols = categoryCols.filter(!onlyGapFeature.contains(_)),
    doNotHashingCateCols = doNotHashingCateCols,
    doNotHashingContCols = doNotHashingContCols,
    params = params,
    useSelector = false,
    featureHasherNumFeature = featureHasherNumFeature,
    featureHashColNm = "feature_hashed_click",
    colNmSuffix = "click"
)

// Gap 모델용 Pipeline
val userDefinedFeatureListForAssemblerGap = Array(
    "app_usage_token_click_cntvec",
    "click_cnt",
    "feature_hashed_click",
    "send_hournum_cd_click_enc"
)

println("Creating Gap transformer pipeline...")
val transformPipelineGap = makePipeline(
    labelCols = Array(),
    indexedFeatureCol = indexedFeatureColGap,
    scaledFeatureCol = scaledFeatureColGap,
    selectedFeatureCol = selectedFeatureColGap,
    tokenCols = Array(),
    vectorCols = Array(),
    continuousCols = Array(),
    categoryCols = Array(),
    doNotHashingCateCols = Array(),
    doNotHashingContCols = Array(),
    params = params,
    useSelector = false,
    featureHasherNumFeature = 0,
    featureHashColNm = "feature_hashed_gap",
    colNmSuffix = "gap",
    userDefinedFeatureListForAssembler = userDefinedFeatureListForAssemblerGap
)

// Pipeline fitting (샘플 데이터로 수행)
println(s"Fitting Click transformer pipeline (sample rate: $pipelineSampleRate)...")
val transformerClick = transformPipelineClick.fit(
    trainDFRev.sample(false, pipelineSampleRate, 42)
)
println("Click transformer fitted successfully")

// Click transformer 적용
println("Transforming training data with Click transformer...")
var transformedTrainDF = transformerClick.transform(trainDFRev)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)
println("Training data transformed with Click pipeline (cached)")

println(s"Fitting Gap transformer pipeline (sample rate: $pipelineSampleRate)...")
val transformerGap = transformPipelineGap.fit(
    transformedTrainDF.sample(false, pipelineSampleRate, 42)
)
println("Gap transformer fitted successfully")

// Gap transformer 적용
println("Transforming training data with Gap transformer...")
transformedTrainDF = transformerGap.transform(transformedTrainDF)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)
println("Training data transformed with Gap pipeline (cached)")

// Test 데이터 변환
println("Transforming test data...")
var transformedTestDF = transformerClick.transform(testDFRev)
transformedTestDF = transformerGap.transform(transformedTestDF)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)
println("Test data transformed (cached)")

// 원본 데이터 unpersist (메모리 확보)
trainDFRev.unpersist()
testDFRev.unpersist()

println("=" * 80)
println("Pipeline transformation completed")
println("=" * 80)


// ===== Paragraph 7: Save Transformers and Transformed Data =====

// =============================================================================
// Transformer 및 Transformed Data 저장
// =============================================================================

println("=" * 80)
println("Saving transformers and transformed data...")
println("=" * 80)

// 1단계: Transformer 저장
val transformerClickPath = s"aos/sto/transformPipelineXDRClick${transformedDataVersion}"
val transformerGapPath = s"aos/sto/transformPipelineXDRGap${transformedDataVersion}"

println(s"Saving Click transformer to: $transformerClickPath")
transformerClick.write.overwrite().save(transformerClickPath)
println("Click transformer saved successfully")

println(s"Saving Gap transformer to: $transformerGapPath")
transformerGap.write.overwrite().save(transformerGapPath)
println("Gap transformer saved successfully")

// 2단계: Train/Test 데이터를 Suffix별 및 Hour별 배치 저장 (메모리 과부하 방지)
// Sliding window 방식으로 suffix 그룹 생성
// - suffixGroupSize: 한 번에 처리할 suffix 개수
// - suffixSlide: Slide 크기
//   * suffixSlide = suffixGroupSize: Overlap 없음 (예: [0,1], [2,3], ...)
//   * suffixSlide < suffixGroupSize: Overlap 있음 (예: [0,1], [1,2], ...)

println(s"Suffix group configuration:")
println(s"  - Group size: $suffixGroupSize")
println(s"  - Slide: $suffixSlide")

// Sliding window로 suffix 그룹 생성
val suffixGroups = (0 to 15).map(_.toHexString).sliding(suffixGroupSize, suffixSlide).toList

println(s"  - Total suffix groups: ${suffixGroups.length}")
suffixGroups.zipWithIndex.foreach { case (suffixGroup, groupIdx) =>
    println(s"    Group ${groupIdx + 1}/${suffixGroups.length}: Suffixes ${suffixGroup.mkString(", ")}")
}
println("=" * 80)

// Hour 그룹 설정 (raw_data_generation.scala와 동일)
val hourGroupSize = 5  // 한 번에 처리할 시간대 개수
val hourSlide = 5      // Slide 크기

println(s"Hour group configuration:")
println(s"  - Hour range: ${hourRange.mkString(", ")}")
println(s"  - Group size: $hourGroupSize")
println(s"  - Slide: $hourSlide")

// Sliding window로 시간대 그룹 생성
val hourGroups = hourRange.sliding(hourGroupSize, hourSlide).toList

println(s"  - Total hour groups: ${hourGroups.length}")
hourGroups.zipWithIndex.foreach { case (hourGroup, groupIdx) =>
    println(s"    Group ${groupIdx + 1}/${hourGroups.length}: Hours ${hourGroup.mkString(", ")}")
}
println("=" * 80)

// Train과 Test 데이터셋 정의
val datasetsToSave = Seq(
    ("training", transformedTrainDF, s"aos/sto/transformedTrainDFXDR${transformedDataVersion}", 20),
    ("test", transformedTestDF, s"aos/sto/transformedTestDFXDF${transformedDataVersion}", 20)
)

// 각 데이터셋에 대해 suffix 그룹별 및 hour 그룹별로 저장
datasetsToSave.foreach { case (datasetName, dataFrame, outputPath, basePartitions) =>
    println(s"Saving transformed $datasetName data by suffix and hour (sliding window)...")
    println(s"  Output path: $outputPath")
    println("=" * 80)

    suffixGroups.zipWithIndex.foreach { case (suffixGroup, suffixGroupIdx) =>
        println(s"  [Suffix Group ${suffixGroupIdx + 1}/${suffixGroups.length}] Processing $datasetName suffixes: ${suffixGroup.mkString(", ")}")
        
        val suffixGroupDF = dataFrame
            .filter(suffixGroup.map(s => s"suffix = '$s'").mkString(" OR "))
        
        // Hour 그룹별로 추가 처리
        hourGroups.zipWithIndex.foreach { case (hourGroup, hourGroupIdx) =>
            println(s"    [Hour Group ${hourGroupIdx + 1}/${hourGroups.length}] Processing hours: ${hourGroup.mkString(", ")}")
            
            val suffixHourGroupDF = suffixGroupDF
                .filter(F.col("send_hournum_cd").isin(hourGroup: _*))
            
            // 데이터가 있는지 확인 (take(1)로 빠르게 체크)
            if (suffixHourGroupDF.take(1).nonEmpty) {
                suffixHourGroupDF
                    .repartition(basePartitions * suffixGroupSize) // 그룹 크기에 비례한 파티션 수
                    .write
                    .mode("overwrite") // Dynamic partition overwrite
                    .option("compression", "snappy")
                    .partitionBy("send_ym", "send_hournum_cd", "suffix")
                    .parquet(outputPath)
                
                println(s"    [Hour Group ${hourGroupIdx + 1}/${hourGroups.length}] $datasetName data saved")
            } else {
                println(s"    [Hour Group ${hourGroupIdx + 1}/${hourGroups.length}] No data, skipped")
            }
        }
        
        println(s"  [Suffix Group ${suffixGroupIdx + 1}/${suffixGroups.length}] Completed for all hours")
    }
    
    println("=" * 80)
    println(s"Transformed $datasetName data saved successfully")
}

println("=" * 80)
println("All transformers and transformed data saved successfully")
println("=" * 80)
println(s"Summary:")
println(s"  - Click Transformer: $transformerClickPath")
println(s"  - Gap Transformer: $transformerGapPath")
println(s"  - Training Data: aos/sto/transformedTrainDFXDR${transformedDataVersion}")
println(s"  - Test Data: aos/sto/transformedTestDFXDF${transformedDataVersion}")
println(s"  - Version: $transformedDataVersion")
println("=" * 80)

// 메모리 정리
transformedTrainDF.unpersist()
transformedTestDF.unpersist()

println("Data transformation pipeline completed!")


// =============================================================================
// End of Data Transformation
// =============================================================================
// 
// 설정 방법 (Paragraph 3):
//   // 테스트 기간만 설정하면 학습 월이 자동 계산됩니다
//   val predictionDTSta = "20251201"  // 테스트 시작일
//   val predictionDTEnd = "20260101"  // 테스트 종료일
//   → sendMonth 자동 계산: "202511" (20251201의 하루 전인 20251130의 월)
//   
//   val period = 6  // 학습 기간 (개월)
//   → sendYmList 자동 생성: ["202506", "202507", ..., "202511"]
//   
//   예시:
//     predictionDTSta = "20260101" → sendMonth = "202512" (6개월: 202507~202512)
//     predictionDTSta = "20251115" → sendMonth = "202511" (6개월: 202506~202511)
//     predictionDTSta = "20250301" → sendMonth = "202502" (6개월: 202509~202502)
// 
// 다음 단계:
// 1. Transformed data를 로딩하여 모델 학습 (train_and_evaluate.scala)
// 
// 확인 방법:
//   // Transformed training data 확인
//   val df = spark.read.parquet(s"aos/sto/transformedTrainDFXDR${transformedDataVersion}")
//   println(s"Training records: ${df.count()}")
//   df.printSchema()
//   
//   // Transformed test data 확인
//   val dfTest = spark.read.parquet(s"aos/sto/transformedTestDFXDF${transformedDataVersion}")
//   println(s"Test records: ${dfTest.count()}")
//   
//   // Feature 확인
//   df.select("indexedFeaturesClick", "indexedFeaturesGap").show(5, false)
// 
// =============================================================================

```


#### Short summary: 

empty definition using pc, found symbol in pc: 