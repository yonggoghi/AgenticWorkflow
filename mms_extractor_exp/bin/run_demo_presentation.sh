#!/bin/bash

# MMS Extractor Presentation Demo 실행 스크립트
#
# Usage:
#   ./bin/run_demo_presentation.sh              # Run presentation app
#   ./bin/run_demo_presentation.sh --generate   # Generate demo data first, then run

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="/Users/yongwook/workspace/AgenticWorkflow/venv/bin/python"
STREAMLIT_PORT=8502

# Check venv python
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python not found at $VENV_PYTHON"
    exit 1
fi

STREAMLIT="$VENV_PYTHON -m streamlit"

# Export PYTHONPATH
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Parse arguments
GENERATE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate)
            GENERATE=true
            shift
            ;;
        --port)
            STREAMLIT_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "MMS Extractor Presentation Demo"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --generate    Generate demo data before running the app"
            echo "  --port PORT   Streamlit port (default: 8502)"
            echo "  -h, --help    Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "  MMS Extractor Presentation Demo"
echo "========================================="

# Generate demo data if requested
if [ "$GENERATE" = true ]; then
    echo ""
    echo "Generating demo data..."
    echo ""
    cd "$PROJECT_DIR"
    $VENV_PYTHON scripts/generate_demo_data.py
    if [ $? -ne 0 ]; then
        echo ""
        echo "Error: Demo data generation failed"
        exit 1
    fi
    echo ""
fi

# Check if demo data exists
DEMO_DIR="$PROJECT_DIR/data/demo_results"
if [ -d "$DEMO_DIR" ] && [ "$(ls -A "$DEMO_DIR"/*.json 2>/dev/null)" ]; then
    NUM_FILES=$(ls "$DEMO_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "Demo data: $NUM_FILES files in $DEMO_DIR"
else
    echo "Warning: No demo data found in $DEMO_DIR"
    echo "  Run with --generate flag to create demo data first:"
    echo "  $0 --generate"
    echo ""
fi

echo ""
echo "Starting Streamlit on port $STREAMLIT_PORT..."
echo "URL: http://localhost:$STREAMLIT_PORT"
echo ""

cd "$PROJECT_DIR"
$STREAMLIT run apps/demo_presentation.py --server.port "$STREAMLIT_PORT"
