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

# Let's import AutoModelForCasualLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Let's choose a small, powerful model suitable for Colab.
# Alternatives you could try (might need login/agreement):
# model_id = "unsloth/gemma-3-4b-it-GGUF"
model_id = "Qwen/Qwen2.5-3B-Instruct"
# model_id = "microsoft/Phi-4-mini-instruct"
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

# Method 2: alternatively, you can create a pipeline that includes your model and tokenizer
# The pipeline wraps tokenization, generation, and decoding

pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = "auto", # Match model dtype
                device_map = "auto" # Ensure pipeline uses the same device mapping
                )


outputs = pipe(prompt,
               max_new_tokens = 1000, # max_new_tokens limits the length of the generated response.
               temperature = 1, # temperature controls randomness (lower = more focused).
               )

# Print the generated text
print_markdown(outputs[0]['generated_text'])

import requests
from pathlib import Path

# --- Get the PDF File ---
pdf_url = "https://s206.q4cdn.com/479360582/files/doc_financials/2025/q1/2025-q1-earnings-transcript.pdf"
pdf_filename = "google_earning_transcript.pdf"
pdf_path = Path(pdf_filename)

# Download the file if it doesn't exist
if not pdf_path.exists():
    response = requests.get(pdf_url)
    response.raise_for_status()  # Check for download errors
    pdf_path.write_bytes(response.content)
    print(f"PDF downloaded successfully to {pdf_path}")
else:
    print(f"PDF file already exists at {pdf_path}")


# --- Read Text from PDF using pypdf ---
pdf_text = ""

print(f"Reading text from {pdf_path}...")
reader = pypdf.PdfReader(pdf_path)
num_pages = len(reader.pages)
print(f"PDF has {num_pages} pages.")

# Extract text from each page
all_pages_text = []
for i, page in enumerate(reader.pages):

    page_text = page.extract_text()
    if page_text:  # Only add if text extraction was successful
        all_pages_text.append(page_text)
    # print(f"Read page {i+1}/{num_pages}") # Uncomment for progress

# Join the text from all pages
pdf_text = "\n".join(all_pages_text)
print(f"Successfully extracted text. Total characters: {len(pdf_text)}")

# Define a limit for the context length to avoid overwhelming the model

MAX_CONTEXT_CHARS = 6000

def answer_question_from_pdf(document_text, question, llm_pipeline):
    """
    Answers a question based on the provided document text using the loaded LLM pipeline.

    Args:
        document_text (str): The text extracted from the PDF.
        question (str): The user's question.
        llm_pipeline (transformers.pipeline): The initialized text-generation pipeline.

    Returns:
        str: The model's generated answer.
    """
    # Truncate context if necessary
    if len(document_text) > MAX_CONTEXT_CHARS:
        print(f"Warning: Document text ({len(document_text)} chars) exceeds limit ({MAX_CONTEXT_CHARS} chars). Truncating.")
        context = document_text[:MAX_CONTEXT_CHARS] + "..."
    else:
        context = document_text

    # Let's define the Prompt Template
    # We instruct the model to use only the provided document.
    # Using a format the model expects (like Phi-3's chat format) can improve results.
    # <|system|> provides context/instructions, <|user|> is the question.
    # Note: Different models might prefer different prompt structures.
    prompt_template = f"""<|system|>
    You are an AI assistant. Answer the following question based *only* on the provided document text. If the answer is not found in the document, say "The document does not contain information on this topic." Do not use any prior knowledge.

    Document Text:
    ---
    {context}
    ---
    <|end|>
    <|user|>
    Question: {question}<|end|>
    <|assistant|>
    Answer:""" # We prompt the model to start generating the answer

    print(f"\n--- Generating Answer for: '{question}' ---")

    # Run Inference on the chosen model
    outputs = llm_pipeline(prompt_template,
                           max_new_tokens = 500,  # Limit answer length
                           do_sample = True,
                           temperature = 0.2,   # Lower temperature for more factual Q&A
                           top_p = 0.9)

    # Let's extract the answer
    # The output includes the full prompt template. We need the text generated *after* it.
    full_generated_text = outputs[0]['generated_text']
    answer_start_index = full_generated_text.find("Answer:") + len("Answer:")
    raw_answer = full_generated_text[answer_start_index:].strip()

    # Sometimes the model might still include parts of the prompt or trail off.
    # Basic cleanup: Find the end-of-sequence token if possible, or just return raw.
    # Phi-3 uses <|end|> or <|im_end|>
    end_token = "<|end|>"
    if end_token in raw_answer:
            raw_answer = raw_answer.split(end_token)[0]

    print("--- Generation Complete ---")
    return raw_answer

# Let's test the function
test_question = "What is this document about?"
generated_answer = answer_question_from_pdf(pdf_text, test_question, pipe)

print("\nTest Question:")
print_markdown(f"**Q:** {test_question}")
print("\nGenerated Answer:")
print_markdown(f"**A:** {generated_answer}")

# Make sure we have the pdf_text
# Configuration: Models available for selection
# Use models known to fit in Colab free tier with 4-bit quantization

available_models = {
    "Llama 3.2": "unsloth/Llama-3.2-3B-Instruct",
    "Microsoft Phi-4 Mini": "microsoft/Phi-4-mini-instruct",
    "Google Gemma 3": "unsloth/gemma-3-4b-it-GGUF"
    }

# --- Global State (or use gr.State in Blocks) ---
# To keep track of the currently loaded model/pipeline
current_model_id = None
current_pipeline = None
print(f"Models available for selection: {list(available_models.keys())}")


# Define a function to Load/Switch Models
def load_llm_model(model_name):
    """Loads the selected LLM, unloading the previous one."""
    global current_model_id, current_pipeline, tokenizer, model

    new_model_id = available_models.get(model_name)
    if not new_model_id:
        return "Invalid model selected.", None  # Return error message and None pipeline

    if new_model_id == current_model_id and current_pipeline is not None:
        print(f"Model {model_name} is already loaded.")
        # Indicate success but don't reload
        return f"{model_name} already loaded.", current_pipeline

    print(f"Switching to model: {model_name} ({new_model_id})...")

    # Unload previous model (important for memory)
    # Clear variables and run garbage collection
    current_pipeline = None
    if "model" in locals():
        del model
    if "tokenizer" in locals():
        del tokenizer
    if "pipe" in locals():
        del pipe
    torch.cuda.empty_cache()  # Clear GPU memory cache
    import gc

    gc.collect()
    print("Previous model unloaded (if any).")

    # --- Load the new model ---
    loading_message = f"Loading {model_name}..."
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(new_model_id, trust_remote_code = True)

        # Load Model (Quantized)
        model = AutoModelForCausalLM.from_pretrained(new_model_id,
                                                     torch_dtype = "auto",  # "torch.float16", # Or bfloat16 if available
                                                     load_in_4bit = True,
                                                     device_map = "auto",
                                                     trust_remote_code = True)

        # Create Pipeline
        loaded_pipeline = pipeline(
            "text-generation", model = model, tokenizer = tokenizer, torch_dtype = "auto", device_map = "auto")

        print(f"Model {model_name} loaded successfully!")
        current_model_id = new_model_id
        current_pipeline = loaded_pipeline  # Update global state
        # Use locals() or return values with gr.State for better Gradio practice
        return f"{model_name} loaded successfully!", loaded_pipeline  # Status message and the pipeline object

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        current_model_id = None
        current_pipeline = None
        return f"Error loading {model_name}: {e}", None  # Error message and None pipeline

# --- Function to handle Q&A Submission ---
# This function now relies on the globally managed 'current_pipeline'
# In a more robust Gradio app, you'd pass the pipeline via gr.State
def handle_submit(question):
    """Handles the user submitting a question."""
    if not current_pipeline:
        return "Error: No model is currently loaded. Please select a model."
    if not pdf_text:
        return "Error: PDF text is not loaded. Please run Section 4."
    if not question:
        return "Please enter a question."

    print(f"Handling submission for question: '{question}' using {current_model_id}")
    # Call the Q&A function defined in Section 5
    answer = answer_question_from_pdf(pdf_text, question, current_pipeline)
    return answer

# --- Build Gradio Interface using Blocks ---
print("Building Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
    # PDF Q&A Bot Using Hugging Face Open-Source Models
    Ask questions about the document ('{pdf_filename}' if loaded, {len(pdf_text)} chars).
    Select an open-source LLM to answer your question.
    **Note:** Switching models takes time as the new model needs to be downloaded and loaded into the GPU.
    """
    )

    # Store the pipeline in Gradio state for better practice (optional for this simple version)
    # llm_pipeline_state = gr.State(None)

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(available_models.keys()),
            label="🤖 Select LLM Model",
            value=list(available_models.keys())[0],  # Default to the first model
        )
        status_textbox = gr.Textbox(label="Model Status", interactive=False)

    question_textbox = gr.Textbox(
        label="❓ Your Question", lines=2, placeholder="Enter your question about the document here..."
    )
    submit_button = gr.Button("Submit Question", variant="primary")
    answer_textbox = gr.Textbox(label="💡 Answer", lines=5, interactive=False)

    # --- Event Handlers ---
    # When the dropdown changes, load the selected model
    model_dropdown.change(
        fn = load_llm_model,
        inputs = [model_dropdown],
        outputs = [status_textbox],  # Update status text. Ideally also update a gr.State for the pipeline
        # outputs=[status_textbox, llm_pipeline_state] # If using gr.State
    )

    # When the button is clicked, call the submit handler
    submit_button.click(
        fn = handle_submit,
        inputs = [question_textbox],
        outputs = [answer_textbox],
        # inputs=[question_textbox, llm_pipeline_state], # Pass state if using it
    )

    # --- Initial Model Load ---
    # Easier: Manually load first model *before* launching Gradio for simplicity here
    initial_model_name = list(available_models.keys())[0]
    print(f"Performing initial load of default model: {initial_model_name}...")
    status, _ = load_llm_model(initial_model_name)
    status_textbox.value = status  # Set initial status
    print("Initial load complete.")


# --- Launch the Gradio App ---
print("Launching Gradio demo...")
demo.launch(debug=True)  # debug=True provides more detailed logs
