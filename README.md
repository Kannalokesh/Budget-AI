# Budget-AI: Indian Union Budget 2026-27 AI Assistant

An advanced RAG (Retrieval-Augmented Generation) application to chat with the Union Budget 2026-27. Featuring Voice Input/Output, Smart TOC filtering, and Voyage AI Reranking.

## Features
- **Voice & Text:** Chat via typing or speaking (powered by Faster-Whisper & gTTS).
- **High Precision RAG:** Uses a 2-stage retrieval (FAISS + Voyage AI Reranker).
- **Smart Filtering:** Automatically ignores Table of Contents for factual queries to improve accuracy.
- **Evaluation:** Verified using RAGAS framework ( 91% Context Precision, 100% Context Recall, 92.50% Faithfullness, 87.91% Answer Relevancy ).


## Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Kannalokesh/Budget-AI.git
   cd Budget-AI
2. **Set up Environment variables**
    ```bash
    create .env file
    OPENAI_API_KEY=your_openai_api_key
    VOYAGE_API_KEY=your_voyage_api_key
3. **Install requirements**
    ```bash
    pip install -r requirements.txt
4. **Data Ingestion and VectorBD creation**
    ```bash
    python ingest.py
5. **Run the app**
    ```bash
    streamlit run app.py
Evaluating using RAGAS Framework
1. **RAGAS evaluation**
    ```bash
    python evaluate_rag.py
2. **Display Report**
    ```bash
    python evaluation_report.ipynb

## LLM + Retrieval Strategy comparison
| # | LLM         | Configuration                                   | Faithfulness | Context Recall | Answer Relevancy | Context Precision |
| - | ----------- | ----------------------------------------------- | ------------ | -------------- | ---------------- | ----------------- |
| 1 | gpt-4o-mini | Without Reranker                                | **97.20%**   | **100%**       | 90.50%           | 65.40%            |
| 2 | gpt-5-nano  | With Reranker                                   | 91.67%       | **100%**       | 88.69%           | 87.50%            |
| 3 | gpt-5-nano  | Smart Filtering + Reranker + Prompt Engineering | 92.50%       | **100%**       | 87.91%           | 90.97%            |
| 4 | gpt-4o-mini | Smart Filtering + Reranker + Prompt Engineering | 91.67%       | **100%**       | **99.09%**       | **95.83%**        |

