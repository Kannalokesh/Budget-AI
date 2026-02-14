import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
# Updated imports to remove deprecation warnings
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
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import voyageai

# 1. SETUP RAG COMPONENTS
api_key = os.getenv("OPENAI_API_KEY")
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("budget_faiss", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG LLM (The one used in our app)
rag_llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key, temperature=0)

# EVALUATOR LLM (The one used by Ragas to judge)
# We set temperature to 1 because your API endpoint is rejecting 0.01
evaluator_llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key, temperature=1)

prompt = ChatPromptTemplate.from_template(
    "Context: {context} \n History: {chat_history} \n Question: {input} \n Answer:"
)
document_chain = create_stuff_documents_chain(rag_llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 2. DEFINE TEST DATASET
test_queries = [
    {
        "question": "What are the seven high-speed rail corridors mentioned as growth connectors?",
        "ground_truth": "The seven corridors are i) Mumbai-Pune, ii) Pune-Hyderabad, iii) Hyderabad-Bengaluru, iv) Hyderabad-Chennai, v) Chennai-Bengaluru, vi) Delhi-Varanasi, and vii) Varanasi-Siliguri."
    },
    {
        "question": "What are the proposals for AYUSH and Ayurveda institutes?",
        "ground_truth": "The proposals include: (i) setting up 3 new All India Institutes of Ayurveda; (ii) upgrading AYUSH pharmacies and Drug Testing Labs; and (iii) upgrading the WHO Global Traditional Medicine Centre in Jamnagar."
    },
    {
        "question": "What is the FAST-DS scheme?",
        "ground_truth": "The Foreign Assets of Small Taxpayers  Disclosure Scheme (FAST - DS), It is proposed to introduce a time-bound scheme for declaration of foreign assets and foreign sourced income for taxpayers involving amounts below certain threshold."
    },
    {
        "question":"What are the three 'kartavyas' that inspired the Budget prepared in Kartavya Bhawan?",
        "ground_truth":"The three kartavyas are: 1) To accelerate and sustain economic growth by enhancing productivity, competitiveness, and building resilience to global dynamics; 2) To fulfil aspirations of the people and build their capacity as partners in India’s prosperity; and 3) Aligned with Sabka Sath, Sabka Vikas, to ensure every family, community, region, and sector has access to resources, amenities, and opportunities."
    }
]

# Reranker on test data

results = []
print("Running RAG with Reranker on test queries...")

for query in test_queries:
    # 1. Get 20 docs (Recall)
    initial_docs = retriever.invoke(query["question"])
    
    # 2. Rerank (Precision)
    doc_texts = [d.page_content for d in initial_docs]
    rerank_results = vo.rerank(
        query=query["question"],
        documents=doc_texts,
        model="rerank-2.5",
        top_k=4
    )
    final_docs = [initial_docs[r.index] for r in rerank_results.results]
    
    # 3. Generate Answer
    response = document_chain.invoke({
        "input": query["question"], 
        "chat_history": "", 
        "context": final_docs
    })
    
    results.append({
        "question": query["question"],
        "answer": response,
        "contexts": [doc.page_content for doc in final_docs],
        "ground_truth": query["ground_truth"]
    })

# 4. PREPARE DATASET
dataset = Dataset.from_list(results)

# 5. PERFORM EVALUATION
print("Evaluating with Ragas using GPT-5-nano ...")

# Initialize metrics with the LLM
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

# 6. DISPLAY RESULTS
df = result.to_pandas()

if not df.empty:
    print("\nRAG Evaluation Report")
    # In newer Ragas, column names might not have 'question' if not mapped, 
    # but to_pandas usually includes it.
    print(df.head())
    
    print("\nAverage Scores")
    print(result)

    # Save report
    df.to_csv("budget_rag_evaluation_report.csv", index=False)
    print("\nResults saved to 'budget_rag_evaluation_report.csv'")
else:
    print("Evaluation failed to produce results.")