
    
    
    
    # create_embeddings.py
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("HuggingFace embeddings initialized successfully.")

# Load test bank
test_bank = pd.read_csv("test.csv")
print("Test bank loaded successfully.")

# Convert test bank to documents for Chroma
documents = []
for _, row in test_bank.iterrows():
    text = f"Test Case ID: {row['Test Case ID']}, Module: {row['Module']}, Description: {row['Test Case Description']}, Steps: {row['Execution Steps']}, Expected Result: {row['Expected Result']}"
    documents.append(text)
print("Test bank converted to documents for Chroma DB.")

# Create Chroma DB with embeddings
try:
    print("Creating Chroma DB...")
    vector_db = Chroma.from_texts(documents, embeddings, persist_directory="./chroma_db")
    print("Chroma DB created successfully.")
except Exception as e:
    print(f"Error creating Chroma DB: {e}")
    