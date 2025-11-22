import json
import os
import time
import random
import numpy as np
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from llm_clients import (
    OpenAIClient, 
    GeminiClient, 
    TogetherClient, 
    AnthropicClient, 
    XAIClient
)

# ────────────── System Prompt Template ──────────────
SYSTEM_PROMPT_TEMPLATE = """Answer the following question based on the provided documents:
Question: {query}
Documents:
{context}
Answer:
"""

# ────────────── Load FinDER Dataset ──────────────
def load_finder_dataset(split: str = "train", num_samples: int = None, seed: int = 42, random_sample: bool = True, filter_ids: List[str] = None):
    """
    Load FinDER dataset from Hugging Face
    
    Args:
        split: Dataset split to load (train only available for FinDER)
        num_samples: Number of samples to load (None for all)
        seed: Random seed for reproducible sampling
        random_sample: If True, randomly sample `num_samples` with the given seed
        filter_ids: List of IDs to filter the dataset by. If provided, num_samples is ignored.
    
    Returns:
        Dataset object
    """
    print(f"Loading FinDER dataset (split: {split})...")
    dataset = load_dataset("Linq-AI-Research/FinDER", split=split)

    if filter_ids:
        print(f"Filtering dataset by {len(filter_ids)} IDs...")
        filter_ids_set = set(filter_ids)
        # Filter dataset to include only samples with IDs in the list
        # Try '_id' first (FinDER dataset usually uses this), then 'id'
        dataset = dataset.filter(lambda x: (x.get('_id') or x.get('id')) in filter_ids_set)
        
        print(f"Filtered dataset size: {len(dataset)}")
    elif num_samples:
        if random_sample:
            # Shuffle deterministically with seed, then take head
            print(f"Randomly sampling {num_samples} examples with seed={seed}")
            dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        else:
            # Deterministic head selection
            dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples")
    return dataset

# ────────────── Helper: Get IDs from CSV ──────────────
def get_ids_from_csv(file_path: str) -> List[str]:
    """
    Read IDs from a CSV file.
    Expects an 'id' column in the CSV.
    """
    import csv
    ids = set()
    try:
        # Use utf-8-sig to handle potential BOM
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            # Normalize headers (strip whitespace)
            if reader.fieldnames:
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            for row in reader:
                if 'id' in row and row['id']:
                    ids.add(row['id'].strip())
    except Exception as e:
        print(f"Error reading ID file {file_path}: {e}")
    
    return list(ids)

# ────────────── Create Prompt ──────────────
def create_prompt(question: str, reference: str) -> str:
    """
    Create prompt using the system template
    
    Args:
        question: The question to ask
        reference: The reference documents/context
    
    Returns:
        Formatted prompt string
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        query=question,
        context=reference
    )

# ────────────── Process Single Sample ──────────────
def process_sample(
    sample_data: tuple,
    model_client,
    question_field: str,
    reference_field: str,
    max_tokens: float = 10
) -> Dict[str, Any]:
    """
    Process a single sample
    
    Args:
        sample_data: Tuple of (index, sample)
        model_client: LLM client instance
        question_field: Field name for question
        reference_field: Field name for reference
        max_tokens: Max tokens for generation (int or float for percentage)
    
    Returns:
        Result dictionary
    """
    i, sample = sample_data
    
    try:
        # Extract question and reference from dataset
        question = sample.get(question_field, "")
        reference = sample.get(reference_field, "")
        
        # Skip if question or reference is empty
        if not question or not reference:
            return None
        
        # Create prompt
        prompt = create_prompt(question, reference)
        
        # Measure latency
        start_time = time.time()
        
        # Get response from LLM
        if isinstance(model_client, TogetherClient):
            response = model_client.get_response(prompt, max_tokens=max_tokens)
        else:
            response = model_client.get_response(prompt)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Create result entry (all keys in lowercase)
        result_entry = {
            "id": sample.get("_id", i),  # Use _id from dataset, fallback to index
            "question": question,
            "reference": reference,
            "response": response,
            "model": model_client.model_id,
            "input_tokens": model_client.last_input_tokens,
            "output_tokens": model_client.last_output_tokens,
            "cost": model_client.last_call_cost,
            "latency": round(latency, 3),
            "ttft": model_client.last_ttft,
            "seq_prob": model_client.last_seq_prob,
            "seq_logprob": model_client.last_seq_logprob
        }
        
        # Add reasoning if available
        if "reasoning" in sample:
            result_entry["reasoning"] = sample["reasoning"]
        
        # Add category if available
        if "category" in sample:
            result_entry["category"] = sample["category"]
        
        # Add ground truth answer if available
        if "answer" in sample:
            result_entry["ground_truth"] = sample["answer"]
        
        return result_entry
        
    except Exception as e:
        error_entry = {
            "id": sample.get("_id", i),  # Use _id from dataset, fallback to index
            "question": sample.get(question_field, ""),
            "reference": sample.get(reference_field, ""),
            "answer": sample.get("answer", ""),
            "response": f"ERROR: {str(e)}",
            "model": model_client.model_id,
            "error": str(e)
        }
        # Add reasoning and category if available
        if "reasoning" in sample:
            error_entry["reasoning"] = sample["reasoning"]
        if "category" in sample:
            error_entry["category"] = sample["category"]
        return error_entry

# ────────────── Run Benchmark ──────────────
def run_benchmark(
    model_client,
    dataset,
    output_file: str,
    question_field: str = "question",
    reference_field: str = "context",
    max_workers: int = 1,
    max_tokens: float = 10
):
    """
    Run benchmark on FinDER dataset and save results to JSON
    
    Args:
        model_client: LLM client instance (OpenAIClient, GeminiClient, etc.)
        dataset: FinDER dataset
        output_file: Path to save JSON results
        question_field: Field name for question in dataset
        reference_field: Field name for reference/context in dataset
        max_workers: Number of parallel workers (1 for sequential)
        max_tokens: Max tokens for generation (int or float for percentage)
    """
    results = []
    total_cost = 0.0
    total_latency = 0.0
    results_lock = Lock()
    
    print(f"\nRunning benchmark with {model_client.model_id}...")
    print(f"Total samples: {len(dataset)}")
    print(f"Max workers: {max_workers}")
    print(f"Max tokens: {max_tokens}")
    print(f"Output file: {output_file}\n")
    
    # Prepare sample data
    sample_data = [(i, sample) for i, sample in enumerate(dataset)]
    
    if max_workers == 1:
        # Sequential processing
        for data in tqdm(sample_data, desc="Processing"):
            result = process_sample(data, model_client, question_field, reference_field, max_tokens)
            if result:
                with results_lock:
                    results.append(result)
                    total_cost += result.get("cost", 0.0)
                    total_latency += result.get("latency", 0.0)
                    
                    # Save intermediate results every 10 samples
                    if len(results) % 10 == 0:
                        save_results(results, output_file, total_cost, total_latency, model_client.model_id)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_sample, data, model_client, question_field, reference_field, max_tokens): data[0]
                for data in sample_data
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_sample), total=len(sample_data), desc="Processing"):
                result = future.result()
                if result:
                    with results_lock:
                        results.append(result)
                        total_cost += result.get("cost", 0.0)
                        total_latency += result.get("latency", 0.0)
                        
                        # Save intermediate results every 10 samples
                        if len(results) % 10 == 0:
                            save_results(results, output_file, total_cost, total_latency, model_client.model_id)
    
    # Sort results by id (from dataset _id field)
    results.sort(key=lambda x: str(x.get("id", "")))
    
    # Save final results
    save_results(results, output_file, total_cost, total_latency, model_client.model_id)
    
    print(f"\n{'='*60}")
    print(f"Benchmark completed!")
    print(f"Total samples processed: {len(results)}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average cost per sample: ${total_cost/len(results):.4f}" if results else "N/A")
    print(f"Total latency: {total_latency:.2f}s")
    print(f"Average latency per sample: {total_latency/len(results):.3f}s" if results else "N/A")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

# ────────────── Save Results ──────────────
def save_results(results: List[Dict[str, Any]], output_file: str, total_cost: float, total_latency: float, model_id: str):
    """
    Save results to JSON file with metadata
    
    Args:
        results: List of result dictionaries
        output_file: Path to save JSON results
        total_cost: Total cost of all API calls
        total_latency: Total latency of all API calls
        model_id: Model identifier
    """
    output_data = {
        "metadata": {
            "model": model_id,
            "total_samples": len(results),
            "total_cost": total_cost,
            "average_cost": total_cost / len(results) if results else 0,
            "total_latency": round(total_latency, 2),
            "average_latency": round(total_latency / len(results), 3) if results else 0
        },
        "results": results
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

# ────────────── Main Function ──────────────
def main():
    """
    Main function to run FinDER benchmark
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FinDER Benchmark with LLM Router")
    parser.add_argument("--api", type=str, default="gemini", 
                       help="API provider: openai, gemini, together, anthropic, xai")
    parser.add_argument("--model-id", type=str, default="gemini-2.5-flash",
                       help="Model ID to use")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of samples to process (None for all)")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for generation")
    parser.add_argument("--max-tokens", type=float, default=10,
                       help="Max tokens for generation (default: 10). If < 1.0, treated as percentage of input tokens.")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split (FinDER only has 'train' split)")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="Number of parallel workers (1 for sequential)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--random-sample", action="store_true",
                       help="Randomly sample num_samples with the given seed (default: take first N)")
    parser.add_argument("--id-file", type=str, default=None,
                       help="CSV file containing 'id' column to filter samples. Overrides num-samples.")
    
    args = parser.parse_args()
    
    # Set global seeds for reproducibility (sampling, numpy ops, hashing)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if we should filter by IDs
    filter_ids = None
    if args.id_file and os.path.exists(args.id_file):
        print(f"Found ID file: {args.id_file}")
        filter_ids = get_ids_from_csv(args.id_file)
    elif os.path.exists("result.csv"):
        print(f"Found result.csv, using IDs from it.")
        filter_ids = get_ids_from_csv("result.csv")

    # Load dataset
    dataset = load_finder_dataset(
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        random_sample=args.random_sample, # Need to pass this
        filter_ids=filter_ids
    )
    
    # Create model client based on API provider
    if args.api == "openai":
        model_client = OpenAIClient(model_id=args.model_id, temperature=args.temperature)
    elif args.api == "gemini":
        model_client = GeminiClient(model_id=args.model_id, temperature=args.temperature)
    elif args.api == "together":
        model_client = TogetherClient(model_id=args.model_id, temperature=args.temperature)
    elif args.api == "anthropic":
        model_client = AnthropicClient(model_id=args.model_id, temperature=args.temperature)
    elif args.api == "xai":
        model_client = XAIClient(model_id=args.model_id, temperature=args.temperature)
    else:
        raise ValueError(f"Unknown API provider: {args.api}")
    
    # Set output file
    output_file = os.path.join(
        args.output_dir, 
        f"finder_results_{model_client.short_model_id}.json"
    )
    
    # Run benchmark
    try:
        run_benchmark(
            model_client=model_client,
            dataset=dataset,
            output_file=output_file,
            question_field="text",  # FinDER dataset uses 'text' field
            reference_field="references",   # FinDER dataset uses 'references' field
            max_workers=args.max_workers,
            max_tokens=args.max_tokens
        )
    except Exception as e:
        print(f"Error running benchmark for {model_client.model_id}: {e}")
        raise

if __name__ == "__main__":
    main()
