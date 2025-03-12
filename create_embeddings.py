
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

#HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("HuggingFace embeddings initialized successfully.")

# test bank
test_bank = pd.read_csv("test.csv")
print("Test bank loaded successfully.")

# Chunking each row into smaller parts
def chunk_row(row):
    return [
        f"Test Case Description: {row['Test Case Description']}",
        f"Execution Steps: {row['Execution Steps']}",
        f"Expected Result: {row['Expected Result']}"
    ]

documents = []
for _, row in test_bank.iterrows():
    documents.extend(chunk_row(row))
print("Test bank converted to documents for Chroma DB.")
print(f"Number of documents: {len(documents)}")


try:
    print("Creating Chroma DB...")
    vector_db = Chroma.from_texts(documents, embeddings, persist_directory="./chroma_db")
    print("Chroma DB created successfully.")
except Exception as e:
    print(f"Error creating Chroma DB: {e}")