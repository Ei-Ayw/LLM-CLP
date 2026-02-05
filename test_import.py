import torch
import transformers
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
from model_deberta_v3_mtl import DebertaV3MTL
print("Imported model")
