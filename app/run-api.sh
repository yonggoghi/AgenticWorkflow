#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Current directory: $(pwd)"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Check required API keys
for var in ANTHROPIC_API_KEY AX_API_KEY; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set. Please set it in your environment or .env file."
        exit 1
    fi
done

# Default paths
DEFAULT_ITEM_DATA_PATH="./info/item_info_250401.csv"
DEFAULT_FEW_SHOT_DATA_PATH="./info/few_shot_data_chat_250414_rd_500.csv"

# Create data directory
mkdir -p data

# Set defaults and validate files
if [ -z "$ITEM_DATA_PATH" ]; then
    export ITEM_DATA_PATH="$DEFAULT_ITEM_DATA_PATH"
    echo "Using default ITEM_DATA_PATH: $ITEM_DATA_PATH"
fi

if [ -z "$FEW_SHOT_DATA_PATH" ]; then
    export FEW_SHOT_DATA_PATH="$DEFAULT_FEW_SHOT_DATA_PATH"
    echo "Using default FEW_SHOT_DATA_PATH: $FEW_SHOT_DATA_PATH"
fi

# Validate files
if [ ! -f "$ITEM_DATA_PATH" ]; then
    echo "Warning: File not found at $ITEM_DATA_PATH"
    ls -l $(dirname "$ITEM_DATA_PATH")
fi

if [ ! -f "$FEW_SHOT_DATA_PATH" ]; then
    echo "Warning: File not found at $FEW_SHOT_DATA_PATH"
    ls -l $(dirname "$FEW_SHOT_DATA_PATH")
fi

# Start the API server
echo "Starting MMS Extraction API server..."
python mms-extraction-api.py
