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

dataset_id = "mohit9999/all_news_finance_sm_1h2023_custom"

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

# Let's choose our smaller DeepSeek model, which is suitable for Google Colab and reasoning tasks.
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Loading model: {model_id}")
print("This may take a few minutes...")


# AutoModelForCausalLM.from_pretrained(...): Loads a pre-trained language model for tasks like text generation (e.g., ChatGPT-like behavior).
# model_id: The name or path of the model you want to load from Hugging Face (like "meta-llama/Llama-3-8b").
# torch_dtype="auto": Automatically chooses the best data type (like float16 or float32) based on your hardware for efficiency.
# load_in_4bit=True: Loads the model in 4-bit precision to save memory and run on limited hardware like free Colab GPUs.
# device_map="auto": Automatically puts the model on the right device (GPU if available, otherwise CPU).
# trust_remote_code=True: Use only with trusted models to avoid security risks.

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype = "auto",
                                             load_in_4bit = True,
                                             device_map = "auto",
                                             trust_remote_code = True)

model.eval()
print("Model loaded successfully in 4-bit!")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True)
print("Tokenizer loaded successfully.")

llm_pipeline = pipeline("text-generation", model = model, tokenizer = tokenizer, torch_dtype = "auto", device_map = "auto")
print("Text generation pipeline created successfully.")

# Let's build the full formatted prompt:
# <|im_start|>user: Marks the beginning of the user's message.
# {test_question}: Inserts your question.
# <|im_end|>: Marks the end of the user’s message.
# <|im_start|>assistant: Signals that the assistant (AI) is about to reply.
# <think>: Encourages the AI to generate internal reasoning or thoughts before answering.

test_question = "Explain what electric cars are in simple terms. Keep the thinking short."
test_prompt = f"<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
print(f"Test prompt:\n{test_prompt}")

# Let's test this beast of a model!
# Generate text using the pipeline
# max_new_tokens: Maximum number of tokens to generate
# do_sample: Enable sampling for more diverse outputs (vs greedy decoding). Enables sampling, meaning the model won’t always pick the highest probability next word — instead, it adds some randomness to make responses more creative and diverse.
# temperature: Controls randomness (0.7 is balanced between creative and focused)
# top_p: Nucleus sampling parameter (0.9 keeps responses on topic while allowing variety). Where the model chooses from the top 90% of likely next words (not just the top one). Helps keep output fluent but varied.

print("Testing model with a simple instruction")

outputs = llm_pipeline(test_prompt,
                       max_new_tokens = 4000,
                       do_sample = True,
                       temperature = 0.7,
                       top_p = 0.9)

print(outputs)

# Function to extract and format the reason and output
def format_model_output(output):

    # Extract the content within <think> tags as Reason
    reason_start = output.find("<think>") + len("<think>")
    reason_end = output.find("</think>")
    reason = output[reason_start:reason_end].strip()

    # Extract the content after </think> as Output
    output_start = reason_end + len("</think>")
    model_output_content = output[output_start:].strip()

    # Format the result
    reason = f"**Reason**:\n{reason}\n"
    output = f"**Output**:\n{model_output_content}"
    return reason, output

import re  # Import regular expressions for parsing

def analyze_news_sentiment(news_text, llm_pipeline):
    """
    Analyzes news sentiment using the LLM, providing reasoning and classification.

    Args:
        news_text (str): The combined title and description of the news.
        llm_pipeline (transformers.pipeline): The initialized text-generation pipeline.

    Returns:
        str: A string containing the reasoning and classification.
    """

    # Define the Prompt Template with specific instructions for the model
    test_question = f"""You are a concise Financial News Analyst.
    Analyze the provided news text. Output ONLY the three requested items below, each on a new line, prefixed with the specified label.

    1.  REASONING: Brief financial reasoning (1-2 points max) for the sentiment.
    2.  SENTIMENT: Choose ONE: Positive, Negative, Neutral.
    3.  TAG: A concise topic tag (1-3 words).

    Do NOT add any other text, greetings, or explanations.

    News Text:
    {news_text}"""

    # Format the prompt according to DeepSeek's expected input format with thinking tags
    test_prompt = f"<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    # Print the prompt for debugging purposes to verify what's being sent to the model
    print(f"Test prompt:\n{test_prompt}")

    # Run the model inference with specific generation parameters
    outputs = llm_pipeline(
        test_prompt,
        max_new_tokens = 4000,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.9)

    # Extract the full generated text and parse it to separate reasoning from classification
    # The format_model_output function likely separates the thinking process from the final answer
    full_output = outputs[0]["generated_text"]
    reason, output = format_model_output(full_output)

    # Return both the reasoning process and the final sentiment classification
    return reason, output

# Print a separator line for clarity in the output
print("\n" + "=" * 30 + " TESTING ANALYSIS " + "=" * 30)


# Select a few random indices from the news dataset to test the analysis function
random_indices = random.sample(range(len(news_dataset)), 3)

# Iterate over each randomly selected index
for index in random_indices:
    # Retrieve the full text of the news item at the current index
    sample_news = news_dataset[index]["full_text"]

    # Analyze the sentiment of the sample news using the sentiment analysis function
    reason, output = analyze_news_sentiment(sample_news, llm_pipeline)

    # Print the analysis result header for the current index
    print(f"\n--- Analysis Result for Index {index} ---")

    # Display the reasoning and output in a formatted markdown style
    print_markdown(f"{reason}\n\n{output}")

    # Print a separator line for better readability between results
    print("-" * 60)

import gradio as gr  # Importing the Gradio library for creating the web interface
import random  # Importing the random library to generate random numbers
import re  # Importing the regular expressions library (not used in this snippet)
from transformers import pipeline  # Importing the pipeline function from transformers for model loading

# --- Gradio Helper Functions ---
def get_random_news():
    """Fetches and returns the full_text of a random news item."""
    if not news_dataset:  # Check if the news dataset is empty
        return "Error: No news items available."  # Return an error message if no news items are found
    random_index = random.randint(0, len(news_dataset) - 1)  # Generate a random index to select a news item
    news_text = news_dataset[random_index]['full_text']  # Retrieve the full text of the news item at the random index
    print(f"Fetched news item at index {random_index}")  # Print the index of the fetched news item
    return news_text  # Return the fetched news text

def perform_analysis(news_text):
    """Triggers analysis on the provided news text."""
    if not news_text or news_text.startswith("Error"):  # Check if the news text is empty or an error message
        return "Error: No news text to analyze."  # Return an error message if no valid news text is provided
    print(f"Analyzing news: {news_text[:50]}...")  # Print the first 50 characters of the news text being analyzed
    reason, output = analyze_news_sentiment(news_text, llm_pipeline)  # Analyze the sentiment of the news text
    return reason, output  # Return the reasoning and output from the analysis

# --- Build Gradio Interface ---
with gr.Blocks(theme=gr.themes.Glass()) as demo:  # Create a Gradio Blocks interface with a glass theme
    gr.Markdown("""
    # DeepSeek Financial News Analyzer
    Fetches a random news item from the dataset.
    Click 'Analyze News' to get sentiment classification and the model's reasoning.

    """)

    with gr.Row():  # Create a row for buttons
        btn_fetch = gr.Button("🔄 Fetch Random News Item")  # Button to fetch a random news item
        btn_analyze = gr.Button("💡 Analyze News", variant="primary")  # Button to analyze the news

    news_display = gr.Textbox(  # Textbox to display the news item
        label="📰 News Text (Title & Description)",  # Label for the textbox
        lines=8,  # Number of lines in the textbox
        interactive=False,  # Make the textbox non-interactive
        placeholder="Click 'Fetch Random News Item' to display news."  # Placeholder text
    )
    # Creates a collapsible panel
    with gr.Accordion("🤖 Model Reason", open=True):  # Accordion for model reasoning
        analysis_display = gr.Markdown()  # Markdown display for the reasoning

    with gr.Accordion("🤖 Model Output", open=True):  # Accordion for model output
        analysis_output = gr.Markdown()  # Markdown display for the output

    # --- Event Handlers ---
    btn_fetch.click(  # Set up click event for the fetch button
        fn=get_random_news,  # Function to call when the button is clicked
        inputs=[],  # No inputs needed
        outputs=[news_display]  # Output to the news display textbox
    )

    btn_analyze.click(  # Set up click event for the analyze button
        fn=perform_analysis,  # Function to call when the button is clicked
        inputs=[news_display],  # Input from the news display textbox
        outputs=[analysis_display, analysis_output]  # Outputs to the reasoning and output displays
    )

    # Load initial news item when the app starts
    demo.load(  # Load function to run when the app starts
        fn=get_random_news,  # Function to call to get a random news item
        inputs=None,  # No inputs needed
        outputs=[news_display]  # Output to the news display textbox
    )

# --- Launch the Gradio App ---
print("Launching Gradio demo...")  # Print message indicating the app is launching
demo.launch(debug=True)  # Launch the Gradio app in debug mode
