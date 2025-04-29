#!/usr/bin/env bash

# Install system-level packages needed by faiss-cpu
apt-get update && apt-get install -y swig cmake g++ libopenblas-dev

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
