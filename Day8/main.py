#!/usr/bin/env python
# This file was generated from the README.org found in this directory

# Let's install and import Pydantic
# In Pydantic, BaseModel is the core class that you use to create data models.
# BaseModel is like a blueprint for structured data. It defines the fields, their types, and automatically gives you data validation and type conversion capabilities
from pydantic import BaseModel

# Let's create a new class named "User"
# BaseModel is a special class from Pydantic that performs validation and parsing
# Inside the class, we will declare name, age, and email along with their expected data types using Python type hints
# Pydantic's role is to validate that name is a str, age is an int, and so on.
# If you pass something wrong (like a string instead of a number), Pydantic raises an error.

class User(BaseModel):
    name: str
    age: int
    email: str

# Import necessary libraries
import os
import google.generativeai as genai
from openai import OpenAI  # Make sure you have the latest openai package (pip install --upgrade openai)
from dotenv import load_dotenv
import json

# Importing type hints that help describe what kind of data your Python functions or classes expect or return.
# List: A list of elements, all usually of the same type.
# Example: List[int] means a list of integers like [1, 2, 3].

# Dict: A dictionary (key-value pairs).
# Example: Dict[str, int] means keys are strings and values are integers like {'a': 1, 'b': 2}.

# Union: Either one type or another.
# Example: Union[int, str] means the value can be an int or a str.

# Optional: Means a value can be the type you expect or None.
# Example: Optional[int] is the same as Union[int, None].

# Any: Anything at all — no restriction on type.
# You can pass an int, string, list, object, etc.
from typing import List, Dict, Union, Optional, Any
from IPython.display import display, Markdown

print("Libraries imported successfully!")

# Load environment variables from the .env file
load_dotenv()

# Fetch API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure the APIs
openai_client = OpenAI(api_key = openai_api_key)
genai.configure(api_key = google_api_key)

# Initialize the Gemini model, choose a suitable model like "gemini-2.0-flash"
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Let's define a Pydantic model called scientist that describes what a valid response should look like
class scientist(BaseModel):
  name: str
  field: str
  known_for: list[str]
  birth_year: int

# Let's define a prompt
prompt = """
Give me a JSON object with details about a famous scientist.
Include the following fields: name, field, known_for, and birth_year.
"""

# Helper function to display markdown nicely 
def print_markdown(text):
    """Displays text as Markdown."""
    display(Markdown(text))

# Let's define a sample resume text
resume_text = """
**Jessica Brown**  
jessica.brown@email.com | (416) 555-7890 | linkedin.com/in/jessicabrown

**Summary**  
Marketing professsional with 2 years of experience assisting in digital campaigns, content creation, and social media activities. Comfortable handling multiple tasks and providing general marketing support.

**Experience**

**Marketing Asssistant | Brewster Coffee Co. | Toronto, ON | Jan 2022 – Present**
- Assisted with digital marketing campaigns including email and social media.
- Created blog posts and social media updates to improve audience engagement.
- Managed social media accounts and grew follower numbers.
- Supported coordination of marketing events.
- Conducted market research and competitor analysis.

**Skils**
- Digital Marketing (SEO basics, Email Marketing)
- Social Media Tools (Hootsuite, Buffer)
- Microsoft Office Suite, Google Workspace
- Basic knowledge of Adobe Photoshop

**Education**  
**Bachelor of Commerce, Marketing** | Ryerson University (now Toronto Metropolitan University), Toronto, ON | May 2021
"""

# Let's define a sample job description text
job_description_text = """
# Job Title: Digital Marketing Specialist

**Company:** BrightWave Digital Agency

**Location:** Toronto, ON

## About Us:
BrightWave Digital Agency creates digital marketing campaigns for a variety of clients. We are looking for a Digital Marketing Specialist to join our team and assist in managing campaigns.

## Responsibilities:
- Assist in planning and executing digital marketing campaigns (SEO, SEM, social media, email).
- Use Google Analytics to measure performance and prepare basic performance reports.
- Support social media management tasks including content scheduling and community engagement.
- Perform keyword research and assist in optimizing content for SEO.
- Work with designers to help coordinate campaign materials.
- Keep informed about current digital marketing trends.

## Qualifications:
- Bachelor's degree in Marketing, Communications, or similar.
- 2+ years of digital marketing experience.
- Familiarity with SEO, SEM, Google Analytics, and social media.
- Ability to interpret basic marketing data.
- Good communication and writing skills.
- Knowledge of CRM systems (e.g., HubSpot) helpful.
- Experience with Adobe Creative Suite is beneficial.
"""

def openai_generate(prompt: str,
                    model: str = "gpt-4o",
                    temperature: float = 0.7,
                    max_tokens: int = 1500,
                    response_format: Optional[dict] = None) -> str | dict:
    """
    Generate text using OpenAI API

    This function sends a prompt to OpenAI's API and returns the generated response.
    It supports both standard text generation and structured parsing with response_format.

    Args:
        prompt (str): The prompt to send to the model, i.e.: your instructions for the AI
        model (str): The OpenAI model to use (default: "gpt-4o")
        temperature (float): Controls randomness, where lower values make output more deterministic
        max_tokens (int): Maximum number of tokens to generate, which limits the response length
        response_format (dict): Optional format specification
        In simple terms, response_format is optional. If the user gives me a dictionary, cool! 
        If they don't give me anything, just assume it's None and keep going."

    Returns:
        str or dict: The generated text or parsed structured data, depending on response_format
    """

    
    try:
        # Standard text generation without a specific response format
        if not response_format:
            response = openai_client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system",
                     "content": "You are a helpful assistant specializing in resume writing and career advice.",
                    },
                    {"role": "user", "content": prompt}],
                temperature = temperature,
                max_tokens = max_tokens)
            
            # Extract just the text content from the response
            return response.choices[0].message.content
        
        # Structured response generation (e.g., JSON format)
        else:
            completion = openai_client.beta.chat.completions.parse(
                model = model,  # Make sure to use a model that supports parse
                messages = [
                    # Same system and user messages as above
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in resume writing and career advice.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature = temperature,
                response_format = response_format)

            # Return the parsed structured output
            return completion.choices[0].message.parsed
            
    except Exception as e:
        # Error handling to prevent crashes
        return f"Error generating text: {e}"

prompt = f"""
Context:
You are a professional resume writer helping a candidate tailor their resume for a specific job opportunity. The resume and job description are provided below.

Instruction:
Enhance the resume to make it more impactful. Focus on:
- Highlighting relevant skills and achievements.
- Using strong action verbs and quantifiable results where possible.
- Rewriting vague or generic bullet points to be specific and results-driven.
- Emphasizing experience and skills most relevant to the job description.
- Reorganizing sections if necessary to better match the job requirements.

Resume:
{resume_text}

Output:
Provide a revised and improved version of the resume that is well-formatted. Only return the updated resume.
"""

# Get response from OpenAI API
openai_output = openai_generate(prompt, temperature = 0.7)

# Display the results
print("#### OpenAI Response:")
print(openai_output)
