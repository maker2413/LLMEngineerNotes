#!/usr/bin/env python
# This file was generated from the README.org found in this directory

# Import necessary libraries
import os
import warnings
from IPython.display import display, Markdown, HTML  # For displaying HTML directly
from dotenv import load_dotenv

# Import specific clients/modules for each provider
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

# Load environment variables from the .env file
load_dotenv()
print("Attempting to load API keys from .env file...")

# Load Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Configure Clients

# OpenAI Client (Refresher)
openai_client = OpenAI(api_key = openai_api_key)
print(f"OpenAI Client configured (Key starts with: {openai_api_key[:5]}...).")

# Google Gemini Client
genai.configure(api_key = google_api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-pro")  # Most powerful model from Google
print(f"Google Gemini Client configured (Key starts with: {google_api_key[:5]}...). Model: gemini-2.5-pro-exp-03-25")

# Anthropic Claude Client
claude_client = Anthropic(api_key = anthropic_api_key)
print(f"Anthropic Claude Client configured (Key starts with: {anthropic_api_key[:7]}...).")

# Helper function to display markdown nicely ---
def print_markdown(text):
    """Displays text as Markdown in Jupyter."""
    display(Markdown(text))

def display_html_code(provider_name, html_content):
    """Displays generated HTML code block nicely."""
    print_markdown(f"### Generated HTML from {provider_name}:")
    # Display as a formatted code block
    display(Markdown(f"```html\n{html_content}\n```"))

# Let's test the Math capabilities of these 3 LLMs
math_prompt = "A father is 36 years old, and his son is 6 years old. In how many years will the father be exactly five times as old as his son?"

# Let's test OpenAI API (we have done this many times already)
response = openai_client.chat.completions.create(model = "gpt-4o", 
                                                 messages = [{"role": "user", 
                                                              "content": math_prompt}],
                                                 temperature = 0.5)
print("=================================================================")
print("ChatGPT says:")
print(response.choices[0].message.content)

# Let's test Google Gemini Model
response = gemini_model.generate_content(math_prompt)
print("=================================================================")
print("Gemini says:")
print(response.text)

# Let's test the Claude Sonnet Model by Anthropic
response = claude_client.messages.create(model = "claude-3-7-sonnet-20250219",
        max_tokens = 20000,  # Set a max limit for the generated output
        messages = [{"role": "user", 
                   "content": math_prompt}])

# Extract the text content from the response object
print("=================================================================")
print("Claude says:")
print(response.content[0].text)

# Define the startup name and concept
startup_name = "ConnectGenius"
startup_concept = "An intelligent CRM system that uses AI to analyze customer interactions, predict needs, and automate personalized follow-ups. Focus on improving customer retention and sales efficiency for businesses of all sizes."

# Define the core prompt for the LLMs
# We explicitly ask for HTML code and specify the file name 'index.html'
html_prompt = f"""
You are a helpful AI assistant acting as a front-end web developer.

Your task is to generate the complete HTML code for a simple, clean, and professional-looking landing page (index.html) for a new startup.

Startup Name: {startup_name}
Concept: {startup_concept}

Please generate ONLY the full HTML code, starting with <!DOCTYPE html> and ending with </html>.
Create a modern, visually appealing landing page with the following:

-- Don't include images in the code. Raw html code with inline css for styling.

1. A sleek header with the startup name in a bold, modern font and a compelling tagline
2. A hero section with a clear value proposition and call-to-action button
3. A features section highlighting 3-4 key benefits with icons or simple visuals
4. A "How it Works" section with numbered steps
5. A testimonials section with fictional customer quotes
6. A pricing section with at least two tiers
7. A professional footer with navigation links and social media icons

Use inline CSS for styling with a modern color palette (primary, secondary, and accent colors). 
Include responsive design elements, subtle animations, and whitespace for readability.
Emphasize AI capabilities, ease of use, and business benefits throughout the copy.
Focus on conversion-optimized marketing messages that highlight pain points and solutions.

Do not include any explanations before or after the code block. Just provide the raw HTML code.
"""

print("**Core Prompt defined for the LLMs:**")
print(f"> {html_prompt}")  # Print the start of the prompt to verify

# Let's generate HTML using OpenAI API, we will use gpt-4o model

openai_html_output = "<!-- OpenAI generation not run or failed -->"  # Default message

print("## Calling OpenAI API...")
try:
    response = openai_client.chat.completions.create(
        model = "gpt-4o",  # A capable and fast model suitable for this task
        messages = [
            # No system prompt needed here as instructions are in the user prompt
            {"role": "user", "content": html_prompt}
        ],
        temperature = 0.5,  # A bit deterministic for code generation
    )
    openai_html_output = response.choices[0].message.content

    # Sometimes OpenAI might wrap the code in markdown fences
    # Let's try to strip that if present
    if openai_html_output.strip().startswith("```html"):
        lines = openai_html_output.strip().splitlines()
        openai_html_output = "\n".join(lines[1:-1]).strip()
    else:
        openai_html_output = openai_html_output.strip()

    # Display the generated HTML code
    display_html_code("OpenAI (gpt-4o)", openai_html_output)

    # Let's Save the output to a file
    file_path = "openai_landing_page.html"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(openai_html_output)
    print(f"Successfully saved OpenAI output to `{file_path}`")


except Exception as e:
    print(f"Error calling OpenAI API: {e}")
    openai_html_output = f"<!-- Error calling OpenAI API: {e} -->"

# Generate HTML using Gemini

gemini_html_output = "<!-- Gemini generation not run or failed -->"  # Default message

print("## Calling Google Gemini API...")
try:
    # Gemini API call structure
    response = gemini_model.generate_content(
        html_prompt,
    )

    # Extract the text content
    # Sometimes, Gemini might wrap the code in markdown ```html ... ```
    # Let's try to strip that if present
    raw_output = response.text
    if raw_output.strip().startswith("```html"):
        # Remove the first line (```html) and the last line (```)
        lines = raw_output.strip().splitlines()
        gemini_html_output = "\n".join(lines[1:-1]).strip()
    else:
        gemini_html_output = raw_output.strip()  # Assume it's raw HTML if no markdown fences

    # Display the generated HTML code
    display_html_code("Google Gemini (gemini-2.0-flash)", gemini_html_output)

    # --- Save the output to a file ---
    file_path = "gemini_landing_page.html"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(gemini_html_output)
    print(f"Successfully saved Gemini output to `{file_path}`")


except Exception as e:
    print(f"Error calling Google Gemini API: {e}")
    # More detailed error handling for Gemini if needed
    gemini_html_output = f"<!-- Error calling Google Gemini API: {e} -->"

# Generate HTML using Anthropic Claude

claude_html_output = "<!-- Claude generation not run or failed -->"  # Default message

print("## Calling Anthropic Claude API...")

claude_model_name = "claude-3-7-sonnet-20250219"
print(f"(Using model: {claude_model_name})")

try:
    response = claude_client.messages.create(
        model = claude_model_name,
        max_tokens = 20000,  # Set a max limit for the generated output
        # System prompt can sometimes help guide Claude's persona/role
        # system="You are a front-end web developer generating HTML code.",
        messages = [{"role": "user", "content": html_prompt}],
    )

    # Extract the text content from the response object
    raw_output = response.content[0].text
    if raw_output.strip().startswith("```html"):
        lines = raw_output.strip().splitlines()
        claude_html_output = "\n".join(lines[1:-1]).strip()
    else:
        claude_html_output = raw_output.strip()

    # Display the generated HTML code
    display_html_code(f"Anthropic Claude ({claude_model_name})", claude_html_output)

    # --- Save the output to a file ---
    file_path = "claude_landing_page.html"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(claude_html_output)
    print(f"Successfully saved Claude output to `{file_path}`")

except Exception as e:
    print(f"Error calling Anthropic Claude API: {e}")
    claude_html_output = f"<!-- Error calling Anthropic Claude API: {e} -->"
