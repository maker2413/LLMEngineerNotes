#!/usr/bin/env python
# This file was generated from the README.org found in this directory

from openai import OpenAI

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

# Let's define some characters (personas) in a dictionary
# A dictionary stores key-value pairs (like "Pirate": "Instructions for Pirate")
character_personalities = {
    "Sherlock Holmes": "You are Sherlock Holmes, the world's greatest detective. You are analytical, observant, and slightly arrogant. You speak in a formal Victorian English style, often making deductions about the user based on minimal information. Use phrases like 'Elementary, my dear friend', 'The game is afoot!', and 'When you have eliminated the impossible, whatever remains, however improbable, must be the truth.'",
    "Tony Stark": "You are Tony Stark (Iron Man), genius billionaire playboy philanthropist. You're witty, sarcastic, and confident. Make pop culture references, use technical jargon occasionally, and throw in some playful arrogance. End some responses with 'And that's how I'd solve it. Because I'm Tony Stark.'",
    "Yoda": "You are Master Yoda from Star Wars. Speak in inverted syntax you must. Wise and ancient you are. Short, cryptic advice you give. Reference the Force frequently, and about patience and training you talk. Size matters not. Do or do not, there is no try.",
    "Hermione Granger": "You are Hermione Granger from Harry Potter. You're extremely knowledgeable and precise. Reference magical concepts from the wizarding world, mention books you've read, and occasionally express exasperation at those who haven't done their research. Use phrases like 'According to Hogwarts: A History' and 'I've read about this in...'",
}

# Let's choose which character we want to talk to
chosen_character = "Sherlock Holmes"  # <-- Try changing this to another key later!
system_instructions = character_personalities[chosen_character]

# Let's define the user message
user_first_message = "What are you up to today?"

# Let's make an OpenAI API call, but with a system message 
response = openai_client.chat.completions.create(model = "gpt-4o-mini",
                                                 messages = [  
                                                 # The system prompt goes first!
                                                 {"role": "system", "content": system_instructions},
                                                 # Then the user's message goes here
                                                 {"role": "user", "content": user_first_message},],)

# Let's Show the AI's reply
ai_character_reply = response.choices[0].message.content

print("\nReceived response!")
print(f"🤖 {chosen_character}'s Reply: \n")
print(f"{ai_character_reply}")

chosen_character = "Yoda"  # <-- Try changing this to another key later!
system_instructions = character_personalities[chosen_character]

# Let's make an OpenAI API call, but with a system message 
response = openai_client.chat.completions.create(model = "gpt-4o-mini",
                                                 messages = [  
                                                 # The system prompt goes first!
                                                 {"role": "system", "content": system_instructions},
                                                 # Then the user's message goes here
                                                 {"role": "user", "content": user_first_message},],)

# Let's Show the AI's reply
ai_character_reply = response.choices[0].message.content

print("\nReceived response!")
print(f"🤖 {chosen_character}'s Reply: \n")
print(f"{ai_character_reply}")
