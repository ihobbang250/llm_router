import json
import os
import glob
import argparse
import csv
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Aggregate seq_prob from result files by ID")
    parser.add_argument("--input-dir", type=str, default="exp_results", help="Directory containing result JSON files")
    parser.add_argument("--output-file", type=str, default="aggregated_seq_probs.json", help="Output file path")
    parser.add_argument("--include-logprob", action="store_true", help="Also include seq_logprob if available")
    parser.add_argument("--score-file", type=str, default="result.csv", help="CSV file containing model scores")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file
    score_file = args.score_file
    
    # Load scores if available
    scores = defaultdict(lambda: defaultdict(dict))
    if os.path.exists(score_file):
        print(f"Loading scores from {score_file}...")
        try:
            with open(score_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'id' not in row or 'metric' not in row:
                        continue
                    sid = row['id']
                    metric = row['metric']
                    for key, value in row.items():
                        if key not in ['id', 'metric']:
                            model = key
                            try:
                                val = float(value)
                            except ValueError:
                                val = value
                            scores[sid][model][metric] = val
        except Exception as e:
            print(f"Error reading score file: {e}")
    
    # Dictionary to store aggregated results
    # Structure: { id: { model_name: seq_prob_list } }
    aggregated_data = defaultdict(dict)
    
    # Find all result files
    search_pattern = os.path.join(input_dir, "finder_results_*.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found matching {search_pattern}")
        # Try 'results' directory if 'exp_results' is empty/default and not found
        if input_dir == "exp_results" and not os.path.exists(input_dir):
             if os.path.exists("results"):
                 print("exp_results not found, trying 'results' directory...")
                 search_pattern = os.path.join("results", "finder_results_*.json")
                 files = glob.glob(search_pattern)

    if not files:
        print(f"No files found.")
        return

    print(f"Found {len(files)} files. Processing...")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine model name from filename
            filename = os.path.basename(file_path)
            model_name = filename.replace("finder_results_", "").replace(".json", "")
            
            print(f"Processing results for model: {model_name}")
            
            for result in data.get("results", []):
                sample_id = result.get("id")
                if sample_id is None:
                    continue
                
                # Ensure ID is string
                sample_id = str(sample_id)
                
                if args.include_logprob:
                    # If including logprob, structure changes to { model: { "prob": [], "logprob": [] } }
                    entry = {}
                    if "seq_prob" in result:
                        entry["seq_prob"] = result["seq_prob"]
                    if "seq_logprob" in result:
                        entry["seq_logprob"] = result["seq_logprob"]
                    
                    if entry:
                        aggregated_data[sample_id][model_name] = entry
                else:
                    # Default: just seq_prob list
                    if "seq_prob" in result:
                        aggregated_data[sample_id][model_name] = result["seq_prob"]
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Merge scores into aggregated data
    if scores:
        print("Merging scores into aggregated data...")
        for sample_id, models_data in aggregated_data.items():
            for model_name, data in models_data.items():
                # Check if we have scores for this sample and model
                if sample_id in scores and model_name in scores[sample_id]:
                    model_scores = scores[sample_id][model_name]
                    
                    # If data is a list (just seq_prob), convert to dict
                    if isinstance(data, list):
                        # Update the reference in the dictionary
                        models_data[model_name] = {"seq_prob": data}
                        data = models_data[model_name]
                    
                    # Now data is a dict, update it with scores
                    data.update(model_scores)

                    # Calculate average score if both metrics exist
                    if 'faithfulness' in data and 'factual_correctness' in data:
                        try:
                            faithfulness = float(data['faithfulness'])
                            factual_correctness = float(data['factual_correctness'])
                            data['score'] = (faithfulness + factual_correctness) / 2.0
                        except (ValueError, TypeError):
                            pass

    # Save to file
    print(f"Saving aggregated results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
    print("Done!")

if __name__ == "__main__":
    main()
