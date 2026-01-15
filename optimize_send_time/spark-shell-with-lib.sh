#!/bin/bash
# Spark Shell with all JARs in lib/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

if [ ! -d "$LIB_DIR" ]; then
    echo "Error: lib directory not found"
    exit 1
fi

JAR_LIST=$(find "$LIB_DIR" -name "*.jar" | tr '\n' ',' | sed 's/,$//')

echo "Starting spark-shell with all optimizer libraries..."
echo "JARs from: $LIB_DIR"
echo ""

spark-shell --jars "$JAR_LIST" "$@"
