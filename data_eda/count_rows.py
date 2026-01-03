import os

data_dir = r"d:/project-work/01_nlp_toxicity_classification/data"
files = ["train.csv", "test.csv", "test_public_expanded.csv", "test_private_expanded.csv", "toxicity_individual_annotations.csv", "identity_individual_annotations.csv"]

print(f"{'Filename':<35} | {'Rows':<15} | {'Size (MB)':<10}")
print("-" * 65)

for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        try:
            with open(path, 'rb') as f_obj:
                # Count lines efficiently
                count = sum(1 for _ in f_obj) - 1
            print(f"{f:<35} | {count:<15,} | {size_mb:<10.2f}")
        except Exception as e:
             print(f"{f:<35} | {'Error':<15} | {size_mb:<10.2f}")
    else:
        print(f"{f:<35} | {'Not Found':<15} | {'-':<10}")
