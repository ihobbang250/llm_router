#!/bin/bash

# ==============================================================================
# Script to run Evaluation
# Usage: ./run_eval.sh <input_file> [evaluator_model] [provider] [output_dir] [max_workers]
# Example: ./run_eval.sh bench_results/finder_results_gpt_oss_120b.json gpt-4o openai eval_results 5
# ==============================================================================

set -e

# Parse arguments
INPUT_FILE=${1:-"bench_results/finder_results_gpt-5.json"}
EVAL_MODEL=${2:-"gemini-2.5-flash"}
PROVIDER=${3:-"gemini"}
OUTPUT_DIR=${4:-"eval_results"}
MAX_WORKERS=${5:-100}

if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required."
    echo "Usage: ./run_eval.sh <input_file> [evaluator_model] [provider] [output_dir] [max_workers]"
    exit 1
fi

EXTRA_ARGS=""
if [ -n "$OUTPUT_DIR" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --output-dir $OUTPUT_DIR"
fi
if [ -n "$MAX_WORKERS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max-workers $MAX_WORKERS"
fi

# Run evaluation
python evaluate_results.py \
    "$INPUT_FILE" \
    --model "$EVAL_MODEL" \
    --provider "$PROVIDER" \
    $EXTRA_ARGS
