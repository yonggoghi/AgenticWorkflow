#!/bin/bash
################################################################################
# 모든 최적화 라이브러리 환경 변수 설정
# 
# 사용법:
#   source setup_all_optimizers.sh
#   또는
#   . setup_all_optimizers.sh
#
# 이 스크립트는 다음을 설정합니다:
#   - ORTOOLS_JARS (OR-Tools)
#   - JMETAL_JARS (jMetal)
#   - ALL_OPTIMIZER_JARS (모든 라이브러리 통합)
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

echo "================================================================================"
echo "Setting up Optimizer Libraries Environment"
echo "================================================================================"
echo ""

# lib 디렉토리 확인
if [ ! -d "$LIB_DIR" ]; then
    echo "✗ lib directory not found: $LIB_DIR"
    echo ""
    echo "Please run: ./download_jmetal.sh"
    return 1 2>/dev/null || exit 1
fi

# OR-Tools 설정
ORTOOLS_JARS=$(find "$LIB_DIR" -name "ortools*.jar" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
if [ -n "$ORTOOLS_JARS" ]; then
    export ORTOOLS_JARS
    echo "✓ OR-Tools JARs found and set"
    echo "  $(echo $ORTOOLS_JARS | tr ',' '\n' | wc -l | xargs) JAR file(s)"
else
    echo "⚠ OR-Tools JARs not found in $LIB_DIR"
    echo "  If you need OR-Tools, please download the JARs to $LIB_DIR/"
fi
echo ""

# jMetal 설정
JMETAL_JARS=$(find "$LIB_DIR" -name "jmetal*.jar" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
if [ -n "$JMETAL_JARS" ]; then
    export JMETAL_JARS
    echo "✓ jMetal JARs found and set"
    echo "  $(echo $JMETAL_JARS | tr ',' '\n' | wc -l | xargs) JAR file(s)"
else
    echo "⚠ jMetal JARs not found in $LIB_DIR"
    echo "  Run: ./download_jmetal.sh"
fi
echo ""

# 통합 변수 생성
if [ -n "$ORTOOLS_JARS" ] && [ -n "$JMETAL_JARS" ]; then
    ALL_OPTIMIZER_JARS="$ORTOOLS_JARS,$JMETAL_JARS"
    export ALL_OPTIMIZER_JARS
    echo "✓ Combined optimizer JARs variable created"
    echo "  ALL_OPTIMIZER_JARS=\$ORTOOLS_JARS,\$JMETAL_JARS"
elif [ -n "$ORTOOLS_JARS" ]; then
    ALL_OPTIMIZER_JARS="$ORTOOLS_JARS"
    export ALL_OPTIMIZER_JARS
    echo "✓ Using OR-Tools only"
    echo "  ALL_OPTIMIZER_JARS=\$ORTOOLS_JARS"
elif [ -n "$JMETAL_JARS" ]; then
    ALL_OPTIMIZER_JARS="$JMETAL_JARS"
    export ALL_OPTIMIZER_JARS
    echo "✓ Using jMetal only"
    echo "  ALL_OPTIMIZER_JARS=\$JMETAL_JARS"
else
    echo "✗ No optimizer JARs found"
    echo ""
    echo "Please install at least one optimizer library:"
    echo "  - jMetal: ./download_jmetal.sh"
    echo "  - OR-Tools: Download JARs to $ORTOOLS_DIR/"
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "================================================================================"
echo "Usage Examples"
echo "================================================================================"
echo ""
echo "Spark Shell with all optimizers:"
echo "  spark-shell --jars \$ALL_OPTIMIZER_JARS -i optimize_ost.scala"
echo ""
echo "Spark Shell with specific library:"
echo "  spark-shell --jars \$JMETAL_JARS -i optimize_ost.scala"
echo "  spark-shell --jars \$ORTOOLS_JARS -i optimize_ost.scala"
echo ""
echo "Convenience script:"
echo "  ./spark-shell-jmetal.sh -i optimize_ost.scala"
echo ""
echo "================================================================================"
