#!/bin/bash

echo ">>> [1/3] Uninstalling conflicting libraries..."
pip uninstall -y transformers sentencepiece protobuf tokenizers

echo ">>> [2/3] Installing stable versions (Force Pure-Python Protobuf)..."
# Installing transformers 4.36.2, sentencepiece 0.1.99, protobuf 3.20.3 (Pure Python), and matched tokenizers
# [Critical Fix] --no-binary=protobuf enforces python implementation to avoid C++ ABI conflicts
pip install transformers==4.36.2 sentencepiece==0.1.99 protobuf==3.20.3 tokenizers==0.15.2 --no-binary=protobuf

echo ">>> [3/3] Verifying environment with debug_crash.py..."
export PYTHONIOENCODING=utf-8
# [Critical Fix] Explicitly tell protobuf to use python backend
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python debug_crash.py
