#!/bin/bash
################################################################################
# jMetal 환경 변수 설정 스크립트
# 
# 사용법:
#   source setup_jmetal_env.sh
#   또는
#   . setup_jmetal_env.sh
#
# 이후:
#   spark-shell --jars $JMETAL_JARS -i optimize_ost.scala
################################################################################

# 현재 스크립트의 디렉토리 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

# JAR 파일들이 있는지 확인
if [ ! -d "$LIB_DIR" ]; then
    echo "Error: lib directory not found: $LIB_DIR"
    echo "Please run ./download_jmetal.sh first"
    return 1 2>/dev/null || exit 1
fi

# jMetal JAR 파일 목록 생성 (콤마로 구분)
JMETAL_JARS=$(find "$LIB_DIR" -name "jmetal*.jar" | tr '\n' ',' | sed 's/,$//')

if [ -z "$JMETAL_JARS" ]; then
    echo "Error: No jMetal JAR files found in $LIB_DIR"
    echo "Please run ./download_jmetal.sh first"
    return 1 2>/dev/null || exit 1
fi

# 환경 변수 export
export JMETAL_JARS

echo "✓ jMetal environment variable set successfully!"
echo ""
echo "JMETAL_JARS=$JMETAL_JARS"
echo ""
echo "You can now use:"
echo "  spark-shell --jars \$JMETAL_JARS"
echo "  spark-shell --jars \$JMETAL_JARS -i optimize_ost.scala"
echo ""

# OR-Tools와 함께 사용하는 경우
if [ -n "$ORTOOLS_JARS" ]; then
    export ALL_OPTIMIZER_JARS="$ORTOOLS_JARS,$JMETAL_JARS"
    echo "✓ Detected ORTOOLS_JARS, combined variable created:"
    echo "  ALL_OPTIMIZER_JARS=\$ORTOOLS_JARS,\$JMETAL_JARS"
    echo ""
    echo "Use with both libraries:"
    echo "  spark-shell --jars \$ALL_OPTIMIZER_JARS -i optimize_ost.scala"
    echo ""
fi
