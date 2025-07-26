#!pip install faiss-cpu

import faiss
import numpy as np

# Create FAISS index
embeddings = embed_model.encode(texts)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve(query, top_k=3):
    q_embed = embed_model.encode([query])
    distances, indices = index.search(np.array(q_embed), top_k)
    return [texts[i] for i in indices[0]]
