import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import voyageai
from test_data import test_queries

# 1. SETUP RAG COMPONENTS
api_key = os.getenv("OPENAI_API_KEY")
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("budget_faiss", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# RAG LLM 
rag_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

# EVALUATOR LLM 
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=1)

prompt = ChatPromptTemplate.from_template(
    "Context: {context} \n History: {chat_history} \n Question: {input} \n Answer:"
)
document_chain = create_stuff_documents_chain(rag_llm, prompt)

# 2. RUN RAG WITH RERANKER & SMART FILTER
results = []
print(f"Running RAG on {len(test_queries)} test queries...")

for i, query in enumerate(test_queries):
    user_q = query["question"]
    print(f"[{i+1}/{len(test_queries)}] Processing: {user_q[:50]}...")

    # A. Initial Retrieval (Recall Stage)
    initial_docs = retriever.invoke(user_q)
    
    # B. SMART FILTER 
    meta_keywords = ["contents", "index", "chapters", "overview", "summary", "topics", "outline"]
    is_meta_query = any(word in user_q.lower() for word in meta_keywords)
    
    candidate_docs = []
    for d in initial_docs:
        page_num = d.metadata.get("page", 0)
        content_upper = d.page_content.upper()
        if not is_meta_query:
            # Skip TOC noise (first 4 pages or high dot density)
            if page_num < 4 or "CONTENTS" in content_upper or d.page_content.count("....") > 5:
                continue 
        candidate_docs.append(d)

    if not candidate_docs:
        candidate_docs = initial_docs[:10]

    # C. Rerank (Precision Stage)
    doc_texts = [d.page_content for d in candidate_docs]
    rerank_results = vo.rerank(
        query=user_q,
        documents=doc_texts,
        model="rerank-2.5",
        top_k=4
    )
    final_docs = [candidate_docs[r.index] for r in rerank_results.results]
    
    # D. Generate Answer
    response = document_chain.invoke({
        "input": user_q, 
        "chat_history": "", 
        "context": final_docs
    })
    
    # Rename keys to match Ragas expectations (user_input, response, retrieved_contexts, reference)
    results.append({
        "user_input": user_q,
        "response": response,
        "retrieved_contexts": [doc.page_content for doc in final_docs],
        "reference": query["ground_truth"]
    })

# 3. PREPARE DATASET
dataset = Dataset.from_list(results)

# 4. PERFORM EVALUATION
print("Evaluating with Ragas (Judge: GPT-4o-mini)...")

# Initialize metrics with the LLM and proper formatting
metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
    ContextPrecision(llm=evaluator_llm),
]

result = evaluate(
    dataset,
    metrics=metrics,
)

# 5. DISPLAY RESULTS
df = result.to_pandas()

if not df.empty:
    print("\n--- RAG Evaluation Report ---")
    # Show the specific columns for clarity
    print(df[['user_input', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])
    
    print("\n--- Final Average Scores ---")
    print(result)

    # Save report
    df.to_csv("budget_rag_evaluation_report.csv", index=False)
    print("\nSuccess! Results saved to 'budget_rag_evaluation_report.csv'")
else:
    print("Evaluation failed to produce results.")