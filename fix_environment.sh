#!/bin/bash

echo ">>> [1/3] Uninstalling conflicting libraries..."
pip uninstall -y transformers sentencepiece protobuf tokenizers

echo ">>> [2/3] Installing stable versions..."
# Installing transformers 4.36.2, sentencepiece 0.1.99, protobuf 3.20.3, and matched tokenizers
pip install transformers==4.36.2 sentencepiece==0.1.99 protobuf==3.20.3 tokenizers==0.15.2

echo ">>> [3/3] Verifying environment with debug_crash.py..."
export PYTHONIOENCODING=utf-8
python debug_crash.py
