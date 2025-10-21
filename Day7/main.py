#!/usr/bin/env python
# This file was generated from the README.org found in this directory

# Let's install and import OpenAI Package
from openai import OpenAI  

# Let's import os, which stands for "Operating System"
import os

# This will be used to load the API key from the .env file
from dotenv import load_dotenv
load_dotenv()

# Get the OpenAI API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Let's configure the OpenAI Client using our key
openai_client = OpenAI(api_key=openai_api_key)

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import RetrievalQAWithSourcesChain

# Define the path to your data file
# Ensure 'eleven_madison_park_data.txt' is in the same folder as this notebook
DATA_FILE_PATH = "eleven_madison_park_data.txt"
print(f"Data file path set to: {DATA_FILE_PATH}")

# Let's load Eleven Madison Park Restaurant data, which has been scraped from their website
# The data is saved in "eleven_madison_park_data.txt", Langchain's TextLoader makes this easy to read
print(f"Attempting to load data from: {DATA_FILE_PATH}")

# Initialize the TextLoader with the file path and specify UTF-8 encoding
# Encoding helps handle various characters correctly
loader = TextLoader(DATA_FILE_PATH, encoding = "utf-8")

# Load the document(s) using TextLoader from LangChain, which loads the entire file as one Document object
raw_documents = loader.load()
print(f"Successfully loaded {len(raw_documents)} document(s).")

# Let's split the document into chunks
print("\nSplitting the loaded document into smaller chunks...")

# Let's initialize the splitter, which tries to split the document on common separators like paragraphs (\n\n),
# sentences (.), and spaces (' ').
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,  # Aim for chunks of about 1000 characters
                                               chunk_overlap = 150,)  # Each chunk overlaps with the previous by 150 characters

# Split the raw document(s) into smaller Document objects (chunks)
documents = text_splitter.split_documents(raw_documents)

# Check if splitting produced any documents
if not documents:
    raise ValueError("Error: Splitting resulted in zero documents. Check the input file and splitter settings.")
print(f"Document split into {len(documents)} chunks.")

# Let's initialize our embeddings model. Note that we will use OpenAI's embedding model 
print("Initializing OpenAI Embeddings model...")

# Create an instance of the OpenAI Embeddings model
# Langchain handles using the API key we loaded earlier
embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)

print("OpenAI Embeddings model initialized.")

# Let's Create ChromaDB Vector Store
print("\nCreating ChromaDB vector store and embedding documents...")

# Now the chunks from 'documents' are being converted to a vector using the 'embeddings' model
# The vectors are then stored as a vector in ChromaDB
# You could add `persist_directory="./my_chroma_db"` to save it to disk
# You will need to specify: (1) The list of chunked Document objects and (2) The embedding model to use
vector_store = Chroma.from_documents(documents = documents, embedding = embeddings)  

# Verify the number of items in the store
vector_count = vector_store._collection.count()
print(f"ChromaDB vector store created with {vector_count} items.")

if vector_count == 0:
    raise ValueError("Vector store creation resulted in 0 items. Check previous steps.")
