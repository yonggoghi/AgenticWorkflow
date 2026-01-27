# lib 디렉토리 구조

모든 최적화 라이브러리 JAR 파일들을 `lib/` 디렉토리에 통합하여 관리합니다.

## 디렉토리 구조

```
optimize_send_time/
├── lib/                           # 모든 JAR 파일 통합 디렉토리
│   ├── jmetal-core-6.1.jar       # jMetal Core
│   ├── jmetal-algorithm-6.1.jar  # jMetal Algorithms
│   ├── jmetal-problem-6.1.jar    # jMetal Problems
│   └── ortools-java-9.4.1874.jar # OR-Tools (선택사항)
│
├── setup_jmetal_env.sh           # jMetal 환경 변수 설정
├── setup_all_optimizers.sh       # 모든 라이브러리 환경 변수 설정
├── download_jmetal.sh            # jMetal 자동 다운로드
├── spark-shell-jmetal.sh         # jMetal 지원 Spark shell
└── spark-shell-with-lib.sh       # 모든 lib JAR 로드
```

## JAR 파일 설치

### jMetal (자동)

```bash
./download_jmetal.sh
```

자동으로 `lib/` 디렉토리에 다운로드됩니다:
- `lib/jmetal-core-6.1.jar`
- `lib/jmetal-algorithm-6.1.jar`
- `lib/jmetal-problem-6.1.jar`

### OR-Tools (수동)

```bash
# 다운로드
wget https://repo1.maven.org/maven2/com/google/ortools/ortools-java/9.4.1874/ortools-java-9.4.1874.jar -P lib/

# 또는 curl 사용
curl -L -o lib/ortools-java-9.4.1874.jar \
  https://repo1.maven.org/maven2/com/google/ortools/ortools-java/9.4.1874/ortools-java-9.4.1874.jar
```

### 기타 라이브러리

다른 최적화 라이브러리도 `lib/` 디렉토리에 추가하면 자동으로 인식됩니다:

```bash
cp /path/to/your-library.jar lib/
```

## 환경 변수

### 자동 설정

```bash
source setup_all_optimizers.sh
```

다음 환경 변수가 자동으로 설정됩니다:
- `$JMETAL_JARS`: lib/jmetal*.jar
- `$ORTOOLS_JARS`: lib/ortools*.jar
- `$ALL_OPTIMIZER_JARS`: 모든 JAR 통합

### 수동 설정

```bash
# 모든 JAR 포함
export ALL_JARS=$(find lib -name "*.jar" | tr '\n' ',' | sed 's/,$//')

# Spark shell 실행
spark-shell --jars $ALL_JARS
```

## 사용 예제

### 방법 1: 환경 변수 사용 (권장)

```bash
# 환경 설정
source setup_all_optimizers.sh

# Spark shell 시작
spark-shell --jars $ALL_OPTIMIZER_JARS -i optimize_ost.scala
```

> **중요**: 위처럼 `-i optimize_ost.scala`로 시작한 세션에서는 같은 파일을 다시 `:load optimize_ost.scala`로 실행하지 마세요.  
> 동일한 `case class`/`object` 재정의가 발생하며 spark-shell이 크래시할 수 있습니다.

### 방법 2: 편의 스크립트 사용

```bash
# 모든 lib JAR 자동 로드
./spark-shell-with-lib.sh -i optimize_ost.scala
```

### 방법 3: 직접 지정

```bash
# lib 디렉토리의 모든 JAR 로드
spark-shell --jars $(find lib -name "*.jar" | tr '\n' ',' | sed 's/,$//')
```

## JAR 파일 관리

### 설치된 JAR 확인

```bash
# 모든 JAR 나열
ls -lh lib/*.jar

# jMetal만
ls -lh lib/jmetal*.jar

# OR-Tools만
ls -lh lib/ortools*.jar

# JAR 개수 확인
ls lib/*.jar | wc -l
```

### JAR 제거

```bash
# 특정 라이브러리 제거
rm lib/jmetal*.jar

# 모두 제거
rm lib/*.jar
```

### JAR 재설치

```bash
# jMetal 재설치
rm lib/jmetal*.jar
./download_jmetal.sh

# OR-Tools 재설치
rm lib/ortools*.jar
wget https://repo1.maven.org/maven2/com/google/ortools/ortools-java/9.4.1874/ortools-java-9.4.1874.jar -P lib/
```

## 장점

### 1. 단일 위치 관리
- 모든 JAR 파일이 한 곳에 있어 관리가 용이
- 새 라이브러리 추가가 간단

### 2. 간단한 경로 설정
```bash
# 이전 (복잡)
spark-shell --jars jmetal_jars/*.jar,ortools_jars/*.jar,other_jars/*.jar

# 현재 (단순)
spark-shell --jars $ALL_OPTIMIZER_JARS
# 또는
spark-shell --jars lib/*.jar
```

### 3. 환경 변수 통합
```bash
# 하나의 스크립트로 모든 라이브러리 설정
source setup_all_optimizers.sh
```

### 4. 버전 관리 용이
```bash
# 버전별 디렉토리 구성 가능
lib/
├── jmetal-6.1/
├── jmetal-6.2/
└── ortools-9.4/
```

## 문제 해결

### lib 디렉토리가 없음

```bash
mkdir -p lib
./download_jmetal.sh
```

### JAR 파일이 인식되지 않음

```bash
# 파일 존재 확인
ls -l lib/*.jar

# 권한 확인
chmod 644 lib/*.jar

# 환경 변수 재설정
source setup_all_optimizers.sh
echo $ALL_OPTIMIZER_JARS
```

### 잘못된 JAR 경로

```bash
# 절대 경로 확인
find $(pwd)/lib -name "*.jar"

# 환경 변수에 절대 경로 사용
export ALL_JARS=$(find $(pwd)/lib -name "*.jar" | tr '\n' ',' | sed 's/,$//')
```

## 모범 사례

1. **자동 다운로드 사용**: `./download_jmetal.sh`로 jMetal 설치
2. **환경 변수 활용**: `source setup_all_optimizers.sh`로 편리하게 사용
3. **영구 설정**: `~/.zshrc`에 환경 변수 설정 추가
4. **버전 관리**: `.gitignore`에 `lib/*.jar` 추가 (선택사항)
5. **문서화**: `lib/README.txt`에 설치된 라이브러리 버전 기록

## 참고 자료

- [환경 변수 설정 가이드](ENV_SETUP_GUIDE.md)
- [jMetal 설정 가이드](JMETAL_SETUP.md)
- [빠른 시작 가이드](QUICK_START_JMETAL.md)
