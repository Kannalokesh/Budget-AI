# ArthaAI: India Budget 2026-27 AI Assistant

An advanced RAG (Retrieval-Augmented Generation) application to chat with the Union Budget 2026-27. Featuring Voice Input/Output, Intent Classifier, and Voyage AI Reranking.

## 🚀 Features
- **Voice & Text:** Chat via typing or speaking (powered by Faster-Whisper & gTTS).
- **High Precision RAG:** Uses a 2-stage retrieval (FAISS + Voyage AI Reranker).
- **Smart Filtering:** Using Intent classifier for Structural vs Factual filtering, to ignore Table of Contents for factual queries to improve accuracy.
- **Evaluation:** Verified using RAGAS framework ( 95.8% Context Precision, 97.2% Context Recall, 89.4% Faithfullness, 95.7% Answer Relevancy ).


## 🛠️ Setup Instructions

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
| 1 | gpt-4o-mini | Smart Filtering + Reranker + Prompt Engineering | **89.41%**   |    **97.22**   |    **95.75**     |  **95.8%**       |

