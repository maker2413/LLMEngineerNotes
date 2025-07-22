#!/usr/bin/env python
from openai import OpenAI

# Let's import "os" module, which stands for "Operating System"
# The os module in Python provides a way to interact with the operating system for things like:
# (1) accessing Environment Variables
# (2) Creating, renaming, and deleting files/folders.
import os

# This will be used to load the API key from the .env file
from dotenv import load_dotenv
load_dotenv()

# Get the OpenAI API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Let's configure the OpenAI Client using our key
openai_client = OpenAI(api_key = openai_api_key)
print("OpenAI client successfully configured.")

# Let's view the first few characters in the key
print("Key begins with:", openai_api_key[:5])

# Let's define the message we want to send as the 'user'
my_message = "Write a Poem to my mom Laila congratulating her for her 74th birthday!"
print(f"Sending message to OpenAI: '{my_message}'")

# Let's make an API call to OpenAI and send our message
response = openai_client.chat.completions.create(model = "gpt-4o-mini",
                                                 messages = [{"role": "user", "content": my_message}])

# Let's obtain the AI's reply from the response object
# The response contains lots of info; we need to dig into it to find the text.
# It usually looks like: response -> choices -> [first choice] -> message -> content
ai_reply_content = response.choices[0].message.content

# Let's print the reply
print("\n🤖 AI's Reply: \n")
print(f"{ai_reply_content}")

my_message = "What is the tallest mountain in the world?"
response = openai_client.chat.completions.create(model = "gpt-4o",
                                                 messages = [{"role": "user", "content": my_message}])
ai_reply_content = response.choices[0].message.content

# Print the reply
print("\n🤖 AI's Reply: \n")
print(f"{ai_reply_content}")
