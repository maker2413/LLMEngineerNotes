#!/usr/bin/env python
# This file was generated from the README.org found in this directory

# Hugging Face Login
import os
from huggingface_hub import login, notebook_login

print("Attempting Hugging Face login...")

notebook_login()
print("Login successful (or token already present)!")

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset  # Import the dataset loading function
import gradio as gr
from IPython.display import display, Markdown
import random  # To pick random news items

# Check for GPU availability
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    torch.set_default_device("cuda")
    print("PyTorch default device set to CUDA (GPU).")
else:
    print("WARNING: No GPU detected. Performance will be very slow.")
    print("Go to Runtime > Change runtime type and select GPU.")

# Helper function for markdown display
def print_markdown(text):
    """Displays text as Markdown."""
    display(Markdown(text))

dataset_id = "PaulAdversarial/all_news_finance_sm_1h2023"

print(f"Loading dataset: {dataset_id}...")

# Load the dataset (will download if not cached)
# We might only need the 'train' split, specify split = 'train' if needed
# The datatype of news_dataset is datasets.Dataset (from the datasets library by Hugging Face).
news_dataset = load_dataset(dataset_id, split = "train")
print("Dataset loaded successfully!")

# Let's prepare the data for the LLM
# We'll combine title and description for the input text
def combine_news_text(example):

    # Handle potential None values gracefully
    title = example.get("title", "") or ""
    description = example.get("description", "") or ""

    # Add a separator for clarity
    return {"full_text": f"Title: {title}\nDescription: {description}"}
