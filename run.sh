#!/bin/bash

# ==============================================================================
# Script to run FinDER Benchmark
# Usage: ./run.sh <api_provider> <model_id> [num_samples] [temperature] [max_workers]
# Example: ./run.sh gemini gemini-2.5-flash 10 0.6 5
# ==============================================================================

set -e

# Parse arguments
API_PROVIDER=${1:-"gemini"}
MODEL_ID=${2:-"gemini-2.5-pro"}
NUM_SAMPLES=${3:-1000}
MAX_WORKERS=${5:-20}

# Run benchmark
python finder_benchmark.py \
    --api "$API_PROVIDER" \
    --model-id "$MODEL_ID" \
    --num-samples "$NUM_SAMPLES" \
    --max-workers "$MAX_WORKERS"