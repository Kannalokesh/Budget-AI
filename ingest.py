import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Ensure data directory exists
pdf_path = "data/budget_2026.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: File not found at {pdf_path}. Please create a 'data' folder and put the PDF there.")
    exit()

print("Loading PDF...")
# 1. Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

# 2. Chunking
# Budget documents usually have tables and dense text. 
# We use a slightly smaller chunk size to ensure specific numbers are captured accurately.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} text chunks.")

print(f"Writing first 10 chunks to chunks.txt...")

with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks[:10]):
        f.write(f"--- CHUNK {i+1} ---\n")
        # Include page number from metadata if available
        page = chunk.metadata.get('page', 'Unknown')
        f.write(f"Source Page: {page + 1}\n") 
        f.write(f"Content:\n{chunk.page_content}\n")
        f.write("-" * 50 + "\n\n")

print("Done! Open 'chunks.txt' to view the content.")
    
# 3. Embeddings
print("Loading Embeddings Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 4. FAISS Vector Store
print("Creating Vector Store...")
db = FAISS.from_documents(chunks, embeddings)

# 5. Save
db.save_local("budget_faiss")

print("Success! 'budget_faiss' index created.")

