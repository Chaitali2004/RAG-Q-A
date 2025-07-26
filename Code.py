import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st

# Title
st.title("RAG Q&A Chatbot - Loan Approval Dataset")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Sample_Submission.csv")

df = load_data()
texts = df.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()

# Initialize Embedding Model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS index
embeddings = embed_model.encode(texts)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Retrieval function
def retrieve(query, top_k=3):
    q_embed = embed_model.encode([query])
    distances, indices = index.search(np.array(q_embed), top_k)
    return [texts[i] for i in indices[0]]

# Generator
generator = pipeline('text2text-generation', model='google/flan-t5-base')

def generate_answer(query):
    context = ' '.join(retrieve(query))
    prompt = f"Answer based on context:\nContext: {context}\nQuestion: {query}\nAnswer:"
    return generator(prompt, max_length=200)[0]['generated_text']

# Streamlit UI
query = st.text_input("Ask a question about loan data:")
if st.button("Get Answer") and query:
    answer = generate_answer(query)
    st.success(f"Answer: {answer}")
