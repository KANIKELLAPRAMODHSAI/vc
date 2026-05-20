# ⚡ VectorHire OS

VectorHire is a high-performance **Semantic Talent Matching Engine** built for enterprise-scale recruitment. It leverages a Retrieval-Augmented Generation (RAG) pipeline to transform unstructured PDF resumes into structured, vectorized profiles for sub-second semantic matching.

## 🏗️ System Architecture



The system follows a decoupled architecture separating the ML inference engine from the user interface:

* **Ingestion Layer:** Parses PDF content using `PyPDF2`, extracts entities using `Groq LLaMA-3.3-70b`, and generates dense embeddings via `Sentence-Transformers`.
* **Storage Layer:** Vectorized profiles are upserted into **Pinecone** with rich metadata, enabling high-dimensional semantic search.
* **Intelligence Layer:** Uses `scikit-learn` for precise cosine similarity ranking and an LLM-based reasoning engine for candidate evaluation and skill-gap analysis.

## 🚀 Tech Stack

* **Backend:** FastAPI, Python 3.12
* **Frontend:** Streamlit (RBAC-enabled Dashboard)
* **AI/ML:** `Sentence-Transformers` (all-MiniLM-L6-v2), `Groq` (LLaMA-3.3-70B), `Scikit-Learn`
* **Database:** Pinecone Vector Database
* **Deployment:** Git-synced CI/CD pipeline

## 🔑 Key Features

1.  **Multi-Role Access Control (RBAC):** Distinct dashboards for Freelancers (Profile Ingestion) and Clients (Predictive Matching).
2.  **Semantic Search:** Moves beyond keyword matching to context-aware talent discovery.
3.  **Human-in-the-Loop:** Built-in RLHF (Reinforcement Learning from Human Feedback) triggers for validating AI-generated matches.
4.  **Performance Observability:** Real-time inference latency monitoring and multi-dimensional scoring matrix.

## 🛠️ Quick Start

### 1. Local Setup
```bash
# Clone the repository
git clone [https://github.com/KANIKELLAPRAMODHSAI/vc.git](https://github.com/KANIKELLAPRAMODHSAI/vc.git)
cd vc

# Setup virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
