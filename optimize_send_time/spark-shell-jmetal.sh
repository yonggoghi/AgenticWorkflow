#!/bin/bash
################################################################################
# Spark Shell with jMetal Libraries
# 
# 사용법:
#   ./spark-shell-jmetal.sh                           # Spark shell 시작
#   ./spark-shell-jmetal.sh -i optimize_ost.scala     # 파일 로드하며 시작
#   ./spark-shell-jmetal.sh -i example_jmetal.scala   # 예제 실행
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 환경 변수가 설정되어 있지 않으면 자동 설정
if [ -z "$JMETAL_JARS" ]; then
    echo "Setting up jMetal environment variables..."
    if [ -f "$SCRIPT_DIR/setup_jmetal_env.sh" ]; then
        source "$SCRIPT_DIR/setup_jmetal_env.sh"
    else
        echo "Error: setup_jmetal_env.sh not found"
        exit 1
    fi
fi

# JAR 파일 확인
if [ -z "$JMETAL_JARS" ]; then
    echo "Error: JMETAL_JARS not set"
    echo "Please run: ./download_jmetal.sh"
    exit 1
fi

echo "================================================================================
Starting Spark Shell with jMetal
================================================================================
"

# JMETAL_JARS만 사용할지, ALL_OPTIMIZER_JARS를 사용할지 확인
if [ -n "$ALL_OPTIMIZER_JARS" ]; then
    JARS_TO_USE="$ALL_OPTIMIZER_JARS"
    echo "Using: OR-Tools + jMetal"
else
    JARS_TO_USE="$JMETAL_JARS"
    echo "Using: jMetal only"
fi

echo ""
echo "JARs: $JARS_TO_USE"
echo ""
echo "================================================================================"
echo ""

# Spark Shell 실행
spark-shell --jars "$JARS_TO_USE" "$@"
