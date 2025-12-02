#!/bin/bash
# Quick start script for prompt evolution system

echo "🚀 Prompt Evolution System - Quick Start"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "❗ Please edit .env and add your OPENAI_API_KEY before running."
    echo "   Then run this script again."
    exit 1
fi

# Check if API key is set
if ! grep -q "LLM_API_KEY=" .env 2>/dev/null || grep -q "LLM_API_KEY=your_api_key_here" .env 2>/dev/null; then
    echo "⚠️  LLM_API_KEY not configured in .env"
    echo "   Please edit .env and add your API key, then run again."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source /Users/yongwook/workspace/AgenticWorkflow/venv/bin/activate

# Check dependencies
echo "🔍 Checking dependencies..."
if ! python3 -c "import langchain" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -q -r requirements.txt
fi

echo "✅ Environment ready!"
echo ""
echo "🔄 Starting prompt evolution with verbose logging..."
echo "   (Press Ctrl+C to interrupt and save progress)"
echo ""

# Run with recommended settings for small dataset
python3 prompt_evolution.py \
    --batch_size 3 \
    --anchor_count 3 \
    --anchor_threshold 0.90 \
    --train_ratio 0.7 \
    --verbose

echo ""
echo "✅ Evolution complete!"
echo ""
echo "📁 Check outputs/ directory for results:"
echo "   - final_prompt_*.txt (evolved prompt)"
echo "   - evolution_log.jsonl (batch decisions)"
echo "   - validation_results.json (metrics)"
