# 환경 변수 설정 가이드

최적화 라이브러리들을 편리하게 사용하기 위한 환경 변수 설정 가이드입니다.

## 개요

`$ORTOOLS_JARS`, `$JMETAL_JARS`처럼 환경 변수를 설정하면 매번 긴 JAR 경로를 입력할 필요 없이 간단하게 사용할 수 있습니다.

## 빠른 시작

### 1. JAR 파일 다운로드

```bash
# jMetal 다운로드
./download_jmetal.sh

# OR-Tools는 별도로 다운로드하여 lib/ 디렉토리에 배치
```

### 2. 환경 변수 설정

```bash
# 모든 라이브러리 설정 (권장)
source setup_all_optimizers.sh

# 또는 개별 설정
source setup_jmetal_env.sh
```

### 3. Spark Shell 실행

```bash
# 모든 라이브러리 사용
spark-shell --jars $ALL_OPTIMIZER_JARS -i optimize_ost.scala

# 특정 라이브러리만 사용
spark-shell --jars $JMETAL_JARS -i optimize_ost.scala
spark-shell --jars $ORTOOLS_JARS -i optimize_ost.scala

# 또는 편의 스크립트 사용
./spark-shell-jmetal.sh -i optimize_ost.scala
```

> **중요**: `-i optimize_ost.scala`로 이미 로드한 세션에서는 같은 파일을 다시 `:load optimize_ost.scala`로 실행하지 마세요.  
> (동일한 `case class`/`object` 재정의가 발생하며 spark-shell이 크래시할 수 있습니다.)

## 환경 변수 종류

### JMETAL_JARS

jMetal 라이브러리 JAR 파일들의 경로 (콤마로 구분)

```bash
export JMETAL_JARS="/path/to/jmetal-core-6.1.jar,/path/to/jmetal-algorithm-6.1.jar,..."
```

### ORTOOLS_JARS

Google OR-Tools 라이브러리 JAR 파일들의 경로 (콤마로 구분)

```bash
export ORTOOLS_JARS="/path/to/ortools-java-9.4.1874.jar,..."
```

### ALL_OPTIMIZER_JARS

모든 최적화 라이브러리를 통합한 변수 (자동 생성)

```bash
export ALL_OPTIMIZER_JARS="$ORTOOLS_JARS,$JMETAL_JARS"
```

## 설정 방법

### 방법 1: 매번 수동 설정

터미널을 열 때마다 실행:

```bash
cd ~/workspace/AgenticWorkflow/optimize_send_time
source setup_all_optimizers.sh
```

### 방법 2: 쉘 프로파일에 추가 (권장)

한 번만 설정하면 자동으로 로드됩니다.

**Bash (~/.bashrc 또는 ~/.bash_profile):**

```bash
# Optimizer Libraries
if [ -f ~/workspace/AgenticWorkflow/optimize_send_time/setup_all_optimizers.sh ]; then
    source ~/workspace/AgenticWorkflow/optimize_send_time/setup_all_optimizers.sh > /dev/null 2>&1
fi
```

**Zsh (~/.zshrc):**

```zsh
# Optimizer Libraries
if [ -f ~/workspace/AgenticWorkflow/optimize_send_time/setup_all_optimizers.sh ]; then
    source ~/workspace/AgenticWorkflow/optimize_send_time/setup_all_optimizers.sh > /dev/null 2>&1
fi
```

설정 후 적용:

```bash
source ~/.zshrc  # 또는 source ~/.bashrc
```

### 방법 3: 별칭(Alias) 사용

자주 사용하는 명령어를 별칭으로 등록:

**~/.zshrc 또는 ~/.bashrc에 추가:**

```bash
# Optimizer aliases
alias setup-optimizers='source ~/workspace/AgenticWorkflow/optimize_send_time/setup_all_optimizers.sh'
alias spark-jmetal='~/workspace/AgenticWorkflow/optimize_send_time/spark-shell-jmetal.sh'
alias spark-opt='spark-shell --jars $ALL_OPTIMIZER_JARS'
```

사용:

```bash
setup-optimizers
spark-jmetal -i optimize_ost.scala
spark-opt -i example_jmetal.scala
```

## 검증 방법

### 환경 변수 확인

```bash
# 모든 변수 출력
echo "JMETAL_JARS: $JMETAL_JARS"
echo "ORTOOLS_JARS: $ORTOOLS_JARS"
echo "ALL_OPTIMIZER_JARS: $ALL_OPTIMIZER_JARS"

# JAR 파일 개수 확인
echo $JMETAL_JARS | tr ',' '\n' | wc -l
echo $ORTOOLS_JARS | tr ',' '\n' | wc -l
```

### JAR 파일 존재 확인

```bash
# jMetal
ls -lh lib/*.jar

# OR-Tools
ls -lh lib/*.jar
```

### Spark에서 테스트

```bash
spark-shell --jars $ALL_OPTIMIZER_JARS
```

```scala
// Scala REPL에서
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder
import com.google.ortools.Loader

println("✓ All libraries loaded successfully!")
```

## 디렉토리 구조

```
optimize_send_time/
├── lib/                       # 모든 JAR 파일들 (통합)
│   ├── jmetal-core-6.1.jar
│   ├── jmetal-algorithm-6.1.jar
│   ├── jmetal-problem-6.1.jar
│   └── ortools-java-9.4.1874.jar  # OR-Tools (별도 다운로드)
├── setup_jmetal_env.sh       # jMetal 환경 변수 설정
├── setup_all_optimizers.sh   # 모든 라이브러리 환경 변수 설정 (권장)
├── download_jmetal.sh        # jMetal 자동 다운로드
├── spark-shell-jmetal.sh     # Spark Shell 편의 스크립트
└── spark-shell-with-lib.sh   # 모든 lib JAR 로드
```

## 사용 예제

### 기본 사용

```bash
# 환경 설정
source setup_all_optimizers.sh

# Spark shell 시작
spark-shell --jars $ALL_OPTIMIZER_JARS -i optimize_ost.scala
```

```scala
// Scala에서
import OptimizeSendTime._

val result = allocateUsersWithJMetalNSGAII(df, capacityPerHour)
```

### 알고리즘별 사용

```bash
# jMetal만 사용
spark-shell --jars $JMETAL_JARS -i optimize_ost.scala

# OR-Tools만 사용
spark-shell --jars $ORTOOLS_JARS -i optimize_ost.scala
```

### 배치 작업

```bash
# 환경 변수를 스크립트에서 사용
cat > run_optimization.sh << 'EOF'
#!/bin/bash
source setup_all_optimizers.sh
spark-submit --jars $ALL_OPTIMIZER_JARS --class OptimizeSendTime optimize_job.scala
EOF

chmod +x run_optimization.sh
./run_optimization.sh
```

## 문제 해결

### 환경 변수가 설정되지 않음

```bash
# 현재 쉘에서 확인
echo $JMETAL_JARS

# 비어있으면 수동 설정
source setup_all_optimizers.sh

# 여전히 비어있으면 JAR 파일 확인
ls -l lib/*.jar
ls -l lib/*.jar
```

### JAR 파일을 찾을 수 없음

```bash
# jMetal 재다운로드
./download_jmetal.sh

# OR-Tools는 수동으로 다운로드
mkdir -p lib
# JAR 파일을 lib/에 복사
```

### 쉘 프로파일에 추가했는데 작동 안 함

```bash
# 설정 파일 재로드
source ~/.zshrc  # 또는 ~/.bashrc

# 경로 확인
which spark-shell
echo $PATH

# 스크립트 실행 권한 확인
ls -l setup_all_optimizers.sh
```

### ClassNotFoundException 발생

```bash
# JAR 경로 확인
echo $ALL_OPTIMIZER_JARS | tr ',' '\n'

# 각 파일 존재 확인
for jar in $(echo $ALL_OPTIMIZER_JARS | tr ',' '\n'); do
  if [ -f "$jar" ]; then
    echo "✓ $jar"
  else
    echo "✗ $jar NOT FOUND"
  fi
done
```

## 고급 설정

### 여러 버전 관리

```bash
# 버전별 환경 변수
export JMETAL_6_1_JARS="/path/to/jmetal-6.1/*.jar"
export JMETAL_6_2_JARS="/path/to/jmetal-6.2/*.jar"

# 원하는 버전 선택
export JMETAL_JARS=$JMETAL_6_1_JARS
```

### 조건부 로딩

```bash
# 특정 프로젝트에서만 로딩
if [ "$PWD" = "$HOME/workspace/AgenticWorkflow/optimize_send_time" ]; then
    source setup_all_optimizers.sh
fi
```

### 로깅

```bash
# 로딩 상태 로그 파일에 기록
source setup_all_optimizers.sh 2>&1 | tee ~/optimizer_setup.log
```

## 참고 자료

- [jMetal 설정 가이드](JMETAL_SETUP.md)
- [빠른 시작 가이드](QUICK_START_JMETAL.md)
- [예제 코드](example_jmetal.scala)
