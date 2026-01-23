import re
import pandas as pd
import os

def parse_log(log_path):
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split by evaluation blocks
    # Note: Using the pattern --- EVALUATING: [Name] ---
    eval_blocks = re.split(r'--- EVALUATING: ([\w_]+) ---', content)
    
    results = []
    
    # eval_blocks[0] is header before first evaluation
    # Then it's Name1, Content1, Name2, Content2...
    for i in range(1, len(eval_blocks), 2):
        model_name = eval_blocks[i]
        block_content = eval_blocks[i+1]
        
        # Regex patterns
        # Best Threshold: 0.50, Best F1: 0.7015
        # PR-AUC: 0.7864
        # Accuracy at Best Threshold: 0.9526
        
        thresh_match = re.search(r'Best Threshold:\s*([\d\.]+)', block_content)
        f1_match = re.search(r'Best F1:\s*([\d\.]+)', block_content)
        prauc_match = re.search(r'PR-AUC:\s*([\d\.]+)', block_content)
        acc_match = re.search(r'Accuracy at Best Threshold:\s*([\d\.]+)', block_content)
        
        results.append({
            'Model': model_name,
            'Best Threshold': thresh_match.group(1) if thresh_match else 'N/A',
            'Best F1': f1_match.group(1) if f1_match else 'N/A',
            'PR-AUC': prauc_match.group(1) if prauc_match else 'N/A',
            'Accuracy': acc_match.group(1) if acc_match else 'N/A'
        })

    if not results:
        print("No results found in log file.")
        return

    df = pd.DataFrame(results)
    print("\n### Experiment Results Extraction")
    print(df.to_markdown(index=False))
    return df

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_FILE = os.path.join(BASE_DIR, "log_experiment.log")
    parse_log(LOG_FILE)
