!pip install chromadb sentence-transformers transformers streamlit gradio

# Step 1: Install dependencies (Already done above)
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# Step 2: Load Dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sample_Submission.csv')

# Convert each row into a single text string
texts = df.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()

# Step 3: Initialize Embedding Model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Create ChromaDB collection
client = chromadb.Client()
collection = client.create_collection("loan_docs")

# Add documents to ChromaDB
for i, text in enumerate(texts):
    collection.add(documents=[text], ids=[str(i)])

# Step 5: Define Retrieval Function
def retrieve(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results['documents'][0]

# Step 6: Load Generator Model (FLAN-T5 for free)
generator = pipeline('text2text-generation', model='google/flan-t5-base')

def generate_answer(query):
    context = ' '.join(retrieve(query))
    prompt = f"Answer based on context:\nContext: {context}\nQuestion: {query}\nAnswer:"
    return generator(prompt, max_length=200)[0]['generated_text']

# Test it
query = "What is the applicant income for Loan_ID LP001002?"
print("Q:", query)
print("A:", generate_answer(query))

