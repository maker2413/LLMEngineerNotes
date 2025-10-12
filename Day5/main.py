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

# Helper function for markdown display
def print_markdown(text):
    """Displays text as Markdown in Colab/Jupyter."""
    display(Markdown(text))

# The pipelines are a great and easy way to use models for inference.
# These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks
# Those tasks include Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering.
from transformers import pipeline

# Load a sentiment classifier model on financial news data
# Check the model here: https://huggingface.co/ProsusAI/finbert
pipe = pipeline(model = "ProsusAI/finbert")
pipe("Apple lost 10 Million dollars today due to US tarrifs")

# Let's explore AutoTokenizer
# A tokenizer converts text into numerical IDs that the model understands
# Check a demo for OpenAI's Tokenizers here: https://platform.openai.com/tokenizer
from transformers import AutoTokenizer

# Load tokenizer for GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Encode text to token IDs
tokens = tokenizer("Hello everyone and welcome to LLM and AI Agents Bootcamp")
print(tokens['input_ids'])

# Let's import AutoModelForCasualLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Let's choose a small, powerful model suitable for Colab.
# Alternatives you could try (might need login/agreement):
# model_id = "unsloth/gemma-3-4b-it-GGUF"
# model_id = "Qwen/Qwen2.5-3B-Instruct"
model_id = "microsoft/Phi-3.5-mini-instruct"
# model_id = "unsloth/Llama-3.2-3B-Instruct"

# Let's load the Tokenizer
# The tokenizer prepares text input for the model
# trust_remote_code=True is sometimes needed for newer models with custom code.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True)
print("Tokenizer loaded successfully.")

# Let's Load the Model with Quantization

print(f"Loading model: {model_id}")
print("This might take a few minutes, especially the first time...")

# Create BitsAndBytesConfig for 4-bit quantization
quantization_config = BitsAndBytesConfig(load_in_4bit = True,
                                         bnb_4bit_compute_dtype = torch.float16,  # or torch.bfloat16 if available
                                         bnb_4bit_quant_type = "nf4",  # normal float 4 quantization
                                         bnb_4bit_use_double_quant = True  # use nested quantization for more efficient memory usage
                                         )

# Load the model with the quantization config
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config = quantization_config,
                                             device_map = "auto",
                                             trust_remote_code = True)

# Let's define a prompt
prompt = "Explain how Electric Vehicles work in a funny way!"

# Method 1: Let's test the model and Tokenizer using the .generate() method!

# Let's encode the input first
inputs = tokenizer(prompt, return_tensors = "pt")

# Then we will generate the output
outputs = model.generate(**inputs, max_new_tokens = 1000, use_cache=False)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print_markdown(response)
