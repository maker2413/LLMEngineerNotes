#!/usr/bin/env python
# This file was generated from the README.org found in this directory

import torch  # PyTorch, the backend for transformers
import pypdf  # For reading PDFs
import gradio as gr  # For building the UI
from IPython.display import display, Markdown  # For nicer printing in notebooks

import os

from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
print("Attempting to load API keys from .env file...")

# Load Keys
hf_token = os.getenv("HF_API_KEY")

# Login
login(token=hf_token)

# Check if GPU is available (essential for running these models)
# Why GPU is Important: LLMs involve billions of calculations (matrix multiplications).
# GPUs are designed for massive parallel processing, making these calculations thousands of times faster than a standard CPU.
# Running these models on a CPU would take an impractically long time (hours for a single answer instead of seconds/minutes).
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    # Set default device to GPU
    torch.set_default_device("cuda")
    print("PyTorch default device set to CUDA (GPU).")
else:
    print("WARNING: No GPU detected. Running these models on CPU will be extremely slow!")
    print("Make sure 'GPU' is selected in Runtime > Change runtime type.")
