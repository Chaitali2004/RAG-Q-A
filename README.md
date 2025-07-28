#  RAG Q&A Chatbot – Loan Approval Dataset

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that intelligently answers questions based on a loan approval dataset.

It combines:
- **Document Retrieval** using **FAISS**
-  **Generative AI** using HuggingFace's `FLAN-T5`
-  **Interactive Chatbot UI** using **Streamlit**

---

##  Dataset Used
- **Kaggle Loan Approval Dataset**
- [Dataset Link](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)

---

## Features
- Ask questions about loan applicants (e.g., income, status, demographics)
- Uses semantic search (vector similarity) for relevant data
- Generates natural language answers with a light-weight LLM

---

##  Tech Stack
- Python
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (for vector search)
- HuggingFace Transformers (`google/flan-t5-base`)
- Streamlit (UI)

---

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-loan-chatbot.git
cd rag-loan-chatbot
