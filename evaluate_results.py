import json
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_clients_bench import OpenAIClient, GeminiClient, AnthropicClient, TogetherClient, XAIClient

# Evaluation Prompt Template
EVAL_PROMPT_TEMPLATE = """You are an expert evaluator for a financial question answering task.
Compare the Model Response with the Ground Truth Answer for the given Question.

Question: {question}
Ground Truth Answer: {answer}
Model Response: {response}

Task:
Determine if the Model Response is correct based on the Ground Truth Answer.
- For numerical answers: Check if the values are equivalent. Allow for formatting differences (e.g., "$3.8 million" vs "3.8", "23.6%" vs "0.236", "1,000" vs "1000").
- For text answers: Check if the semantic meaning is substantially the same.
- If the Model Response contains the correct answer but also extra information, count it as correct unless the extra information contradicts the answer.

Output strictly valid JSON with the following format:
{{
    "correct": boolean,
    "reason": "brief explanation of why it is correct or incorrect"
}}
"""

def evaluate_sample(client, sample):
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    response = sample.get("response", "")
    
    if not answer:
        # If there is no ground truth, we can't evaluate automatically
        return {"correct": False, "reason": "No ground truth answer provided"}

    prompt = EVAL_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        response=response
    )
    
    try:
        eval_response = client.get_response(prompt)
        # Clean up response to ensure it's valid JSON (sometimes models add markdown blocks)
        cleaned_response = eval_response.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_response)
        return result
    except Exception as e:
        return {"correct": False, "reason": f"Evaluation failed: {str(e)}"}

def evaluate_file(input_file, output_file, evaluator_model="gpt-4o", api_provider="openai", max_workers=1):
    print(f"Loading results from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    
    results = data.get("results", [])
    if not results:
        print("No results found in file.")
        return

    # Initialize evaluator client
    if api_provider == "openai":
        client = OpenAIClient(model_id=evaluator_model)
    elif api_provider == "gemini":
        client = GeminiClient(model_id=evaluator_model)
    elif api_provider == "anthropic":
        client = AnthropicClient(model_id=evaluator_model)
    elif api_provider == "together":
        client = TogetherClient(model_id=evaluator_model)
    elif api_provider == "xai":
        client = XAIClient(model_id=evaluator_model)
    else:
        raise ValueError(f"Unsupported provider: {api_provider}")

    print(f"Evaluating {len(results)} samples using {evaluator_model} with {max_workers} workers...")
    
    # Parallel processing
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(evaluate_sample, client, sample): sample 
                for sample in results
            }
            
            for future in tqdm(as_completed(future_to_sample), total=len(results), desc="Evaluating"):
                sample = future_to_sample[future]
                try:
                    eval_result = future.result()
                    sample["evaluation"] = eval_result
                except Exception as e:
                    sample["evaluation"] = {"correct": False, "reason": f"Execution failed: {str(e)}"}
    else:
        for sample in tqdm(results, desc="Evaluating"):
            eval_result = evaluate_sample(client, sample)
            sample["evaluation"] = eval_result

    # Calculate accuracy
    correct_count = sum(1 for s in results if s.get("evaluation", {}).get("correct"))
    evaluated_results = results
    
    # Update metadata
    accuracy = correct_count / len(results) if results else 0
    if "metadata" not in data:
        data["metadata"] = {}
    
    data["metadata"]["evaluation_model"] = evaluator_model
    data["metadata"]["accuracy"] = accuracy
    data["results"] = evaluated_results
    
    print(f"Evaluation complete. Accuracy: {accuracy:.2%}")
    
    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluated results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Benchmark Results with LLM")
    parser.add_argument("input_file", help="Path to the input JSON results file")
    parser.add_argument("--output-file", help="Path to save evaluated results (default: input_file_evaluated.json)")
    parser.add_argument("--output-dir", help="Directory to save evaluated results")
    parser.add_argument("--model", default="gpt-4o", help="Evaluator model ID (default: gpt-4o)")
    parser.add_argument("--provider", default="openai", help="API provider for evaluator (default: openai)")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    
    args = parser.parse_args()
    
    output_file = args.output_file
    if not output_file:
        if args.output_dir:
            base_name = os.path.basename(args.input_file)
            name, ext = os.path.splitext(base_name)
            output_file = os.path.join(args.output_dir, f"{name}_evaluated{ext}")
        else:
            base, ext = os.path.splitext(args.input_file)
            output_file = f"{base}_evaluated{ext}"
        
    evaluate_file(args.input_file, output_file, args.model, args.provider, args.max_workers)
