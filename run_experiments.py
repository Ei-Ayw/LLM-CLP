import os
import subprocess
import sys

# Set Hugging Face cache directory and mirror endpoint
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Use generic python command (assumes active environment or correctly linked python)
PYTHON_EXE = "python"
SRC_DIR = "src_script"
RES_DIR = "src_result"
DATA_DIR = "data"

def run_cmd(cmd):
    # Cross-platform command running
    cmd_str = " ".join(cmd)
    print(f"\n>>> Running: {cmd_str}")
    result = subprocess.run(cmd_str, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd_str}")

def main():
    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)

    # --- [1] DATA PREPROCESS ---
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "data_preprocess.py")])

    # --- [2] BASELINE MODELS ---
    # Group 1: Classical (TF-IDF + LR)
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_tfidf_lr.py"), "--mode", "train"])

    # Group 2: Non-Pretrained Deep (BERT+CNN-BiLSTM)
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_baselines.py")])

    # Group 3: Transformer Baselines
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_transformer_baselines.py"), "--model_name", "bert-base-uncased"])
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_transformer_baselines.py"), "--model_name", "roberta-base"])
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_transformer_baselines.py"), "--model_name", "microsoft/deberta-base"])

    # --- [3] OUR PROPOSED MODEL (Stage 1 & 2) ---
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_stage1.py")])
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_stage2_reweight.py")])

    # --- [4] ABLATION STUDIES ---
    # Ablation 1: Pooling (compare Stage 1 result with no_pooling)
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_ablations.py"), "--ablation", "no_pooling"])
    
    # Ablation 2: MTL (compare Stage 1 result with no_mtl)
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "train_ablations.py"), "--ablation", "no_mtl"])
    
    # Ablation 3: Reweight (compare Stage 1 @ w=1 vs Stage 2 @ w=2.5)
    # This comparison uses res_stage1_best.pth vs res_stage2_final.pth

    # --- [5] EVALUATION & COMPARISON ---
    test_suite = [
        # Main Results
        {"name": "Proposed_DeBERTa_V3_MTL_Reweight", "cp": f"{RES_DIR}/res_stage2_final.pth", "type": "deberta_mtl"},
        
        # Baselines
        {"name": "Baseline_BERT_CNN_BiLSTM", "cp": f"{RES_DIR}/res_baseline_bert_cnn_epoch3.pth", "type": "bert_cnn"},
        {"name": "Baseline_BERT_Base", "cp": f"{RES_DIR}/res_baseline_bert_base_uncased.pth", "type": "bert_base"},
        {"name": "Baseline_RoBERTa_Base", "cp": f"{RES_DIR}/res_baseline_roberta_base.pth", "type": "roberta_base"},
        {"name": "Baseline_DeBERTa_V1_Base", "cp": f"{RES_DIR}/res_baseline_microsoft_deberta_base.pth", "type": "deberta_base"},
        
        # Ablations
        {"name": "Ablation_CLS_Only", "cp": f"{RES_DIR}/res_ablation_no_pooling.pth", "type": "deberta_cls"},
        {"name": "Ablation_Single_Task", "cp": f"{RES_DIR}/res_ablation_no_mtl.pth", "type": "deberta_mtl"},
        {"name": "Ablation_No_Reweight", "cp": f"{RES_DIR}/res_stage1_best.pth", "type": "deberta_mtl"},
    ]

    for test in test_suite:
        if os.path.exists(test["cp"]):
            print(f"\n--- EVALUATING: {test['name']} ---")
            run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "eval_threshold.py"), 
                     "--checkpoint", test["cp"], "--model_type", test["type"], 
                     "--output_name", f"metrics_f1_{test['name']}.csv"])
            run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "eval_fairness.py"), 
                     "--checkpoint", test["cp"], "--model_type", test["type"], 
                     "--output_name", f"metrics_fair_{test['name']}.csv"])

    # Final Summary Plot
    run_cmd([PYTHON_EXE, os.path.join(SRC_DIR, "viz_results.py")])

if __name__ == "__main__":
    main()
