# 🇮🇳 India Budget 2026 AI Assistant

An advanced RAG (Retrieval-Augmented Generation) application to chat with the Union Budget 2026-27. Featuring Voice Input/Output, Smart TOC filtering, and Voyage AI Reranking.

## 🚀 Features
- **Voice & Text:** Chat via typing or speaking (powered by Faster-Whisper & gTTS).
- **High Precision RAG:** Uses a 2-stage retrieval (FAISS + Voyage AI Reranker).
- **Smart Filtering:** Automatically ignores Table of Contents for factual queries to improve accuracy.
- **Evaluation:** Verified using RAGAS framework (91% Context Precision, 100% Context Recall, 92.50% Faithfullness, 87.91% Answer Relevancy).

## 🛠️ Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Kannalokesh/Budget-AI.git
   cd Budget-AI