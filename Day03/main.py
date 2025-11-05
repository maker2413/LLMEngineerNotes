#!/usr/bin/env python
# This file was generated from the README.org found in this directory

import os
from IPython.display import display, Markdown
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

print("OpenAI API Key loaded successfully.")
# Let's view the first few characters to confirm it's loaded (DO NOT print the full key)
print(f"Key starts with: {openai_api_key[:5]}...")

# Configure the OpenAI Client using the loaded key
openai_client = OpenAI(api_key=openai_api_key)
print("OpenAI client configured.")

# Define a helper function to display markdown nicely
def print_markdown(text):
    """Displays text as Markdown in Jupyter."""
    display(Markdown(text))

# Let's define the Python function to get a response from the AI Tutor
def get_ai_tutor_response(user_question):
    """
    Sends a question to the OpenAI API, asking it to respond as an AI Tutor.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        str: The AI's response, or an error message.
    """
    # Define the system prompt - instructions for the AI's personality and role
    system_prompt = "You are a helpful and patient AI Tutor. Explain concepts clearly and concisely."

    try:
        # Make the API call to OpenAI
        response = openai_client.chat.completions.create(
            model = "gpt-4o-mini",  # A fast and capable model suitable for tutoring
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
            temperature = 0.7,  # Allows for some creativity but keeps responses focused
        )
        # Extract the answer content
        ai_response = response.choices[0].message.content
        return ai_response

    except Exception as e:
        # Handle potential errors during the API call
        print(f"An error occurred: {e}")
        return f"Sorry, I encountered an error trying to get an answer: {e}"

import gradio as gr

# Let's create a new function that streams the response
def stream_ai_tutor_response(user_question):
    """
    Sends a question to the OpenAI API and streams the response as a generator.

    Args:
        user_question (str): The question asked by the user.

    Yields:
        str: Chunks of the AI's response.
    """

    system_prompt = "You are a helpful and patient AI Tutor. Explain concepts clearly and concisely."

    try:
        # Note: stream = True is the key change here!
        stream = openai_client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
            temperature = 0.7,
            stream = True,  # Enable streaming (magic happens here)
        )

        # Iterate through the response chunks
        full_response = ""  # Keep track of the full response if needed later

        # Loop through each chunk of the response as it arrives
        for chunk in stream:
            # Check if this chunk contains actual text content
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                # Extract the text from this chunk
                text_chunk = chunk.choices[0].delta.content
                # Add this chunk to our growing response
                full_response += text_chunk
                # 'yield' is special - it sends the current state of the response to Gradio
                # This makes the text appear to be typing in real-time
                yield full_response

    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        yield f"Sorry, I encountered an error: {e}"

# Define the mapping for explanation levels
explanation_levels = {
    1: "like I'm 5 years old",
    2: "like I'm 10 years old",
    3: "like a high school student",
    4: "like a college student",
    5: "like an expert in the field",
}

# Create a new function that accepts question and level and streams the response
def stream_ai_tutor_response_with_level(user_question, explanation_level_value):
    """
    Streams AI Tutor response based on user question and selected explanation level.

    Args:
        user_question (str): The question from the user.
        explanation_level_value (int): The value from the slider (1-5).

    Yields:
        str: Chunks of the AI's response.
    """

    # Get the descriptive text for the chosen level
    level_description = explanation_levels.get(
        explanation_level_value, "clearly and concisely"
    )  # Default if level not found

    # Construct the system prompt dynamically based on the level
    system_prompt = f"You are a helpful AI Tutor. Explain the following concept {level_description}."

    print(f"DEBUG: Using System Prompt: '{system_prompt}'")  # For checking

    try:
        stream = openai_client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
            temperature = 0.7,
            stream = True,
        )

        # Iterate through the response chunks
        full_response = ""  # Keep track of the full response if needed later

        # Loop through each chunk of the response as it arrives
        for chunk in stream:
            # Check if this chunk contains actual text content
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                # Extract the text from this chunk
                text_chunk = chunk.choices[0].delta.content
                # Add this chunk to our growing response
                full_response += text_chunk
                # 'yield' is special - it sends the current state of the response to Gradio
                # This makes the text appear to be typing in real-time
                yield full_response

    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        yield f"Sorry, I encountered an error: {e}"

# Define the Gradio interface with both Textbox and slider inputs
ai_tutor_interface_slider = gr.Interface(fn = stream_ai_tutor_response_with_level,  # Function now takes 2 args
    inputs=[
        gr.Textbox(lines = 3, placeholder = "Ask the AI Tutor a question...", label = "Your Question"),
        gr.Slider(
            minimum = 1,
            maximum = 5,
            step = 1,  # Only allow whole numbers
            value = 3,  # Default level (high school)
            label = "Explanation Level",  # Label for the slider
        ),
    ],
    outputs = gr.Markdown(label = "AI Tutor's Explanation (Streaming)", container = True, height = 250),
    title = "🎓 Advanced AI Tutor",
    description = "Ask a question and select the desired level of explanation using the slider.",
    allow_flagging = "never",
)

# Launch the advanced interface
print("Launching Advanced Gradio Interface with Slider...")
ai_tutor_interface_slider.launch()
