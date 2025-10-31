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

# Prompt to analyze the resume against the job description

def analyze_resume_against_job_description(job_description_text: str, resume_text: str, model: str = "openai") -> str:
    """
    Analyze the resume against the job description and return a structured comparison.

    Args:
        job_description_text (str): The job description text.
        resume_text (str): The candidate's resume text.
        model (str): The model to use for analysis ("openai" or "gemini").

    Returns:
        str: A clear, structured comparison of the resume and job description.
    """
    # This prompt instructs the AI to act as a career advisor and analyze how well the resume matches the job description
    # It asks for a structured analysis with 4 specific sections: requirements, matches, gaps, and strengths
    prompt = f"""
    Context:
    You are a career advisor and resume expert. Your task is to analyze a candidate's resume against a specific job description to assess alignment and identify areas for improvement.

    Instruction:
    Review the provided Job Description and Resume. Identify key skills, experiences, and qualifications in the Job Description and compare them to what's present in the Resume. Provide a structured analysis with the following sections:
    1. **Key Requirements from Job Description:** List the main skills, experiences, and qualifications sought by the employer.
    2. **Relevant Experience in Resume:** List the skills and experiences from the resume that match or align closely with the job requirements.
    3. **Gaps/Mismatches:** Identify important skills or qualifications from the Job Description that are missing, unclear, or underrepresented in the Resume.
    4. **Potential Strengths:** Highlight any valuable skills, experiences, or accomplishments in the resume that are not explicitly requested in the job description but could strengthen the application.

    Job Description:

    {job_description_text}

    Resume:

    {resume_text}

    Output:
    Return a clear, structured comparison with the four sections outlined above.
    """

    # This conditional block selects which AI model to use based on the 'model' parameter
    if model == "openai":
        # Uses OpenAI's model to generate the gap analysis with moderate creativity (temperature=0.7)
        gap_analysis = openai_generate(prompt, temperature=0.7)
    elif model == "gemini":
        # Uses Google's Gemini model with less creativity (temperature=0.5) for more focused results
        gap_analysis = gemini_generate(prompt, temperature=0.5)
    else:
        # Raises an error if an invalid model name is provided
        raise ValueError(f"Invalid model: {model}")

    # Returns the generated gap analysis text
    return gap_analysis

# Define Pydantic models for structured output
# The ResumeOutput class is a Pydantic model that defines the structure of the output
# for the resume generation function. It includes two fields:
# (1) updated_resume: A string that contains the final rewritten resume.
# (2) diff_markdown: A string containing the resume's HTML-coloured version highlighting additions and deletions.

class ResumeOutput(BaseModel):
    updated_resume: str
    diff_markdown: str


def generate_resume(
    job_description_text: str, resume_text: str, gap_analysis_openai: str, model: str = "openai") -> dict:
    """
    Generate a tailored resume using OpenAI or Gemini.

    Args:
        job_description_text (str): The job description text.
        resume_text (str): The candidate's resume text.
        gap_analysis_openai (str): The gap analysis result from OpenAI.
        model (str): The model to use for resume generation.

    Returns:
        dict: A dictionary containing the updated resume and diff markdown.
    """
    # Construct the prompt for the AI model to generate the tailored resume.
    # The prompt includes context, instructions, and input data (original resume,
    # target job description, and gap analysis).
    prompt = (
        """
    ### Context:
    You are an expert resume writer and editor. Your goal is to rewrite the original resume to match the target job description, using the provided tailoring suggestions and analysis.

    ---

    ### Instruction:
    1. Rewrite the entire resume to best match the **Target Job Description** and **Gap Analysis to the Job Description**.
    2. Improve clarity, add job-relevant keywords, and quantify achievements.
    3. Specifically address the gaps identified in the analysis by:
       - Adding missing skills and technologies mentioned in the job description
       - Reframing experience to highlight relevant accomplishments
       - Strengthening sections that were identified as weak in the analysis
    4. Prioritize addressing the most critical gaps first
    5. Incorporate industry-specific terminology from the job description
    6. Ensure all quantifiable achievements are properly highlighted with metrics
    7. Return two versions of the resume:
        - `updated_resume`: The final rewritten resume (as plain text)
        - `diff_html`: A version of the resume with inline highlights using color:
            - Additions or rewritten content should be **green**:  
            `<span style="color:green">your added or changed text</span>`
            - Removed content should be **red and struck through**:  
            `<span style="color:red;text-decoration:line-through">removed text</span>`
            - Leave unchanged lines unmarked.
        - Keep all section headers and formatting consistent with the original resume.

    ---

    ### Output Format:

    ```json
    {
    "updated_resume": "<full rewritten resume as plain text>",
    "diff_markdown": "<HTML-colored version of the resume highlighting additions and deletions>"
    }
    ```
    ---
    ### Input:

    **Original Resume:**

    """
        + resume_text
        + """


    **Target Job Description:**

    """
        + job_description_text
        + """


    **Analysis of Resume vs. Job Description:**

    """
        + gap_analysis_openai
    )

    # Depending on the selected model, call the appropriate function to generate the resume.
    # If the OpenAI model is selected, it uses a temperature of 0.7 for creativity.
    if model == "openai":
        updated_resume_json = openai_generate(prompt, temperature = 0.7, response_format = ResumeOutput)
    # If the Gemini model is selected, it uses a lower temperature of 0.5 for more focused results.
    elif model == "gemini":
        updated_resume_json = gemini_generate(prompt, temperature = 0.5)
    else:
        # Raise an error if an invalid model name is provided.
        raise ValueError(f"Invalid model: {model}")

    # Return the generated resume output as a dictionary.
    return updated_resume_json

# Define Pydantic models for structured output
# The CoverLetterOutput class is a Pydantic model that defines the structure of the output for the cover letter generation.
# It ensures that the output will contain a single field, 'cover_letter', which is a string.

class CoverLetterOutput(BaseModel):
    cover_letter: str

# The generate_cover_letter function creates a cover letter based on the provided job description and updated resume.
# It takes three parameters:
# (1) job_description_text: A string containing the job description for the position.
# (2) updated_resume: A string containing the candidate's updated resume.
# (3) model: A string indicating which model to use for generating the cover letter (default is "openai").
# The function returns a dictionary containing the generated cover letter.

def generate_cover_letter(job_description_text: str, updated_resume: str, model: str = "openai") -> dict:
    """
    Generate a cover letter using OpenAI or Gemini.

    Args:
        job_description_text (str): The job description text.
        updated_resume (str): The candidate's updated resume text.
        model (str): The model to use for cover letter generation.

    Returns:
        dict: A dictionary containing the cover letter.
    """

    # Construct the prompt for the AI model, including context and instructions for writing the cover letter.
    prompt = (
        """
    ### Context:
    You are a professional career coach and expert cover letter writer.

    ---

    ### Instruction:
    Write a compelling, personalized cover letter based on the **Updated Resume** and the **Target Job Description**. The letter should:
    1. Be addressed generically (e.g., "Dear Hiring Manager")
    2. Be no longer than 4 paragraphs
    3. Highlight key achievements and experiences from the updated resume
    4. Align with the responsibilities and qualifications in the job description
    5. Reflect the applicant's enthusiasm and fit for the role
    6. End with a confident and polite closing statement

    ---

    ### Output Format (JSON):
    ```json
    {
    "cover_letter": "<final cover letter text>"
    }
    ```
    ---

    ### Input:

    **Updated Resume:**

    """
        + updated_resume
        + """
    **Target Job Description:**

    """
        + job_description_text
    )

    # Depending on the selected model, call the appropriate function to generate the cover letter.
    if model == "openai":
        # Get response from OpenAI API
        updated_cover_letter = openai_generate(prompt, temperature=0.7, response_format=CoverLetterOutput)
    elif model == "gemini":
        # Get response from Gemini API
        updated_cover_letter = gemini_generate(prompt, temperature=0.5)
    else:
        # Raise an error if an invalid model name is provided.
        raise ValueError(f"Invalid model: {model}")

    # Return the generated cover letter as a dictionary.
    return updated_cover_letter

def run_resume_rocket(resume_text: str, job_description_text: str) -> tuple[str, str]:
    """
    Run the resume rocket workflow.

    Args:
        resume_text (str): The candidate's resume text.
        job_description_text (str): The job description text.

    Returns:
        tuple: A tuple containing the updated resume and cover letter.
    """
    # Analyze the candidate's resume against the job description using OpenAI's model.
    # This function will return a structured analysis highlighting gaps and strengths.
    gap_analysis_openai = analyze_resume_against_job_description(job_description_text, 
                                                                 resume_text, 
                                                                 model="openai")

    # Display the gap analysis results in Markdown format for better readability.
    print(gap_analysis_openai)

    # Print separators for clarity in the output.
    print("\n--------------------------------")
    print("--------------------------------\n")

    # Generate an updated resume based on the job description, original resume, and gap analysis.
    # This function will return a JSON-like object containing the updated resume and a diff markdown.
    updated_resume_json = generate_resume(job_description_text, 
                                          resume_text, 
                                          gap_analysis_openai, 
                                          model = "openai")

    # Display the diff markdown which shows the changes made to the resume.
    print(updated_resume_json.diff_markdown)

    # Print separators for clarity in the output.
    print("\n--------------------------------")
    print("--------------------------------\n")

    # Display the updated resume in Markdown format.
    print(updated_resume_json.updated_resume)

    # Print separators for clarity in the output.
    print("\n--------------------------------")
    print("--------------------------------\n")

    # Generate a cover letter based on the job description and the updated resume.
    # This function will return the generated cover letter.
    updated_cover_letter = generate_cover_letter(
        job_description_text, updated_resume_json.updated_resume, model="openai"
    )

    # Display the generated cover letter in Markdown format.
    print(updated_cover_letter.cover_letter)

    # Print separators for clarity in the output.
    print("\n--------------------------------")
    print("--------------------------------\n")

    # Return the updated resume and the generated cover letter as a tuple.
    return updated_resume_json.updated_resume, updated_cover_letter.cover_letter

# Call the run_resume_rocket function with the provided resume and job description texts.
resume, cover_letter = run_resume_rocket(resume_text, job_description_text)

print(resume)

print(cover_letter)
