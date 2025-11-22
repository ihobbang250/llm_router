import json
import os
import pandas as pd
import glob
import argparse

def get_model_name(full_name):
    """
    Extracts short model name from full model ID.
    Logic:
    1. If '/' exists, take the part after the first '/'.
    2. Split the resulting string by '-'.
    3. Join the first two parts with '-'.
    
    Example: Qwen/Qwen3-235B-A22B -> Qwen3-235B
    """
    # Handle slash
    if "/" in full_name:
        name_part = full_name.split("/", 1)[1]
    else:
        name_part = full_name
    
    # Handle hyphens: take first two parts
    parts = name_part.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return name_part

def create_summary_csv(input_dir, output_file):
    print(f"Scanning JSON files in {input_dir}...")
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print("No JSON files found.")
        return

    all_data = {} # id -> {model_name: correct}
    questions = {} # id -> question text
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            model_full_name = data.get("metadata", {}).get("model", "unknown")
            short_name = get_model_name(model_full_name)
            
            results = data.get("results", [])
            print(f"Processing {os.path.basename(file_path)} (Model: {short_name}, Samples: {len(results)})")
            
            for res in results:
                sample_id = res.get("id")
                question = res.get("question", "")
                
                # Check if evaluation exists and is correct
                eval_data = res.get("evaluation", {})
                if isinstance(eval_data, dict):
                    is_correct = eval_data.get("correct", False)
                else:
                    is_correct = False
                
                if sample_id not in all_data:
                    all_data[sample_id] = {}
                
                all_data[sample_id][short_name] = is_correct
                
                if sample_id not in questions:
                    questions[sample_id] = question
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # Convert to DataFrame
    rows = []
    for sample_id, models_data in all_data.items():
        row = {
            "id": sample_id,
            "query": questions.get(sample_id, "")
        }
        row.update(models_data)
        rows.append(row)
        
    if not rows:
        print("No data extracted.")
        return

    df = pd.DataFrame(rows)
    
    # Sort columns (id, query, then models alphabetically)
    model_cols = sorted([c for c in df.columns if c not in ["id", "query"]])
    cols = ["id", "query"] + model_cols
    df = df[cols]
    
    # Sort rows by id
    # Try to sort numerically if ids are like 'finqa0', 'finqa1'
    try:
        df["sort_key"] = df["id"].apply(lambda x: int(x.replace("finqa", "")) if isinstance(x, str) and x.startswith("finqa") and x[5:].isdigit() else x)
        df = df.sort_values("sort_key").drop(columns=["sort_key"])
    except:
        df = df.sort_values("id")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"\nSaved summary to {output_file}")
    print(f"Total queries: {len(df)}")
    print(f"Models included: {', '.join(cols[1:])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Summary CSV from Evaluation Results")
    parser.add_argument("--input-dir", default="eval_results", help="Directory containing evaluated JSON files")
    parser.add_argument("--output-file", default="model_comparison.csv", help="Path to output CSV file")
    
    args = parser.parse_args()
    
    create_summary_csv(args.input_dir, args.output_file)
