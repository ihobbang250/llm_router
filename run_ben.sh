#!/bin/bash

# ==============================================================================
# Script to run FinDER Benchmark
# Usage: ./run.sh <api_provider> <model_id> [num_samples] [temperature] [max_workers]
# Example: ./run.sh gemini gemini-2.5-flash 10 0.6 5
# ==============================================================================

set -e

# Parse arguments
API_PROVIDER=${1:-"openai"}
MODEL_ID=${2:-"gpt-5"}
NUM_SAMPLES=${3:-1000}
MAX_WORKERS=${4:-40}

# Run benchmark
python benchmark.py \
    --api "$API_PROVIDER" \
    --model-id "$MODEL_ID" \
    --num-samples "$NUM_SAMPLES" \
    --max-workers "$MAX_WORKERS" \
    --output-dir "bench_results"