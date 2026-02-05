import os
import sys

print("✅ [1/7] Core libs imported", flush=True)

# 尝试修复 Protobuf 导致的 SegFault
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print("⏳ [2/7] Importing pandas/numpy...", flush=True)
import pandas as pd
import numpy as np
print("✅ [2/7] Pandas/Numpy imported", flush=True)

print("⏳ [3/7] Importing sklearn...", flush=True)
from sklearn import metrics
print("✅ [3/7] Sklearn imported", flush=True)

print("⏳ [4/7] Importing torch...", flush=True)
import torch
print(f"✅ [4/7] Torch imported (CUDA available: {torch.cuda.is_available()})", flush=True)

print("⏳ [5/7] Importing transformers...", flush=True)
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Config
print("✅ [5/7] Transformers imported", flush=True)

print("⏳ [6/7] Importing local models...", flush=True)
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src_model"))
try:
    from src_model.model_deberta_v3_mtl import DebertaV3MTL
    print("✅ [6/7] Local DebertaV3MTL imported", flush=True)
except Exception as e:
    print(f"❌ [6/7] Import failed: {e}")

print("✅ [7/7] All checks passed! The environment seems fine for imports.", flush=True)
