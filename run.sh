#!/bin/bash

# ==============================================================================
# Script to run FinDER Benchmark
# Usage: ./run.sh <api_provider> <model_id> [num_samples] [temperature] [max_workers]
# Example: ./run.sh gemini gemini-2.5-flash 10 0.6 5
# ==============================================================================

set -e

# Parse arguments
API_PROVIDER=${1:-"together"}
MODEL_ID=${2:-"openai/gpt-oss-120b"}
NUM_SAMPLES=${3:-5}
MAX_TOKENS=${4:-50}
MAX_WORKERS=${5:-30}

# Run benchmark
python finder_benchmark.py \
    --api "$API_PROVIDER" \
    --model-id "$MODEL_ID" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --max-workers "$MAX_WORKERS" \
    --output-dir "exp_results"