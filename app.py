import os
import re
import uuid
import tempfile
import voyageai
import streamlit as st

# LangChain & OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Voice
from gtts import gTTS
from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder

# --- LOAD ENVIRONMENT VARIABLES ---
api_key = os.getenv("OPENAI_API_KEY")
voyage_key = os.getenv("VOYAGE_API_KEY")

if voyage_key:
    vo = voyageai.Client(api_key=voyage_key)
else:
    st.error("VOYAGE_API_KEY missing!")

LLM_MODEL = "gpt-4o-mini"

# --- PAGE CONFIG ---
st.set_page_config(page_title="BudgetAI", page_icon="🇮🇳", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
    .block-container { padding-bottom: 1rem; }
    .stMarkdown p { margin-bottom: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Session Controls")
    if st.button("Clear Chat "):
        st.session_state.messages = []

    if st.button("Clear Memory"):
        st.session_state.memory = []

    if st.button("Clear Chat & Memory"):
        st.session_state.messages = []
        st.session_state.memory = []

st.title("🇮🇳 BudgetAI")

# --- 1. UTILITIES ---

def clean_text_for_speech(text: str) -> str:
    if not text: 
        return ""
    replacements = [
        (r'\bi\)', "one."), (r'\bii\)', "two."), (r'\biii\)', "three."),
        (r'\biv\)', "four."), (r'\bv\)', "five."), (r'\bvi\)', "six."),
        (r'\bvii\)', "seven."), (r'\bviii\)', "eight."), (r'\bix\)', "nine."), (r'\bx\)', "ten.")
    ]
    clean_text = text
    for pattern, replacement in replacements:
        clean_text = re.sub(pattern, replacement, clean_text, flags=re.IGNORECASE)
    return clean_text

# --- 2. INTENT CLASSIFIER ---

def classify_intent(query: str):
    """Uses LLM to determine if query is asking for structure/overview or specific facts."""
    if not api_key: 
        return "FACTUAL"
    
    classifier_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    
    prompt = f"""
    Classify the user query into one of two categories:
    1. 'STRUCTURAL': User is asking for the Table of Contents, index, list of topics, chapters, or an overview of what is in the document.
    2. 'FACTUAL': User is asking for specific data, tax rates, scheme details, or allocations.

    Query: "{query}"
    
    Return only the word 'STRUCTURAL' or 'FACTUAL'.
    Category:"""
    
    try:
        response = classifier_llm.invoke(prompt)
        intent = response.content.strip().upper()
        return intent if intent in ["STRUCTURAL", "FACTUAL"] else "FACTUAL"
    except:
        return "FACTUAL"

# --- 3. LOAD MODELS ---

@st.cache_resource
def load_local_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    try:
        vectorstore = FAISS.load_local("budget_faiss", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    except:
        st.error("FAISS Index not found!")
        return None, None
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    return retriever, whisper

retriever, whisper = load_local_models()

# --- 4. RAG COMPONENTS ---

def get_rag_components():
    if not api_key: 
        return None, None
    llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key, temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """
        SYSTEM ROLE:
        You are the highly accurate Official Union Budget 2026 Assistant. 

        STRICT CONSTRAINTS:
        1. SOURCE ONLY: Use ONLY the provided Context to answer. Never use general knowledge.
        2. ACCURACY: If not in Context, state clearly that it is not available.
        3. NO HALLUCINATION: If a fact is not explicitly stated, do not mention it at all.

        CONVERSATION HISTORY: {chat_history}
        CONTEXT: {context}
        USER QUESTION: {input}
        ASSISTANT ANSWER:
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain, retriever

# --- 5. SESSION STATE & DISPLAY ---

if "messages" not in st.session_state: st.session_state.messages = []
if "memory" not in st.session_state: st.session_state.memory = []

def get_formatted_history():
    history_text = ""
    for user_msg, ai_msg in st.session_state.memory[-3:]:
        history_text += f"User: {user_msg}\nAssistant: {ai_msg}\n"
    return history_text

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message: st.audio(message["audio"])
        if "sources" in message:
            with st.expander("📚 View Budget Source Chunks"):
                for i, doc in enumerate(message["sources"]):
                    page_num = doc.metadata.get("page", "N/A")
                    st.markdown(f"**Chunk {i+1} (Page {page_num + 1}):**")
                    st.caption(doc.page_content)
                    st.divider()

# --- 6. INPUT HANDLING ---

final_query = None

if api_key and voyage_key:
    st.write("---")
    col1, col2 = st.columns([1, 8])
    with col1:
        audio_data = mic_recorder(start_prompt="🎙️ Speak", stop_prompt="🛑 Stop", just_once=True, key='recorder')

    user_input = st.chat_input("Ask about Budget 2026...")

    if audio_data and audio_data['bytes']:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(audio_data['bytes'])
            fp_name = fp.name
        with st.spinner("Transcribing..."):
            segments, _ = whisper.transcribe(fp_name, language="en", vad_filter=True)
            final_query = " ".join(seg.text for seg in segments)
        os.remove(fp_name)

    if user_input: 
        final_query = user_input

# --- 7. GENERATE RESPONSE ---

if final_query:
    with st.chat_message("user"):
        st.markdown(final_query)
    st.session_state.messages.append({"role": "user", "content": final_query})

    with st.spinner("Processing..."):
        history_str = get_formatted_history()
        
        # A. Classify Intent using LLM
        intent = classify_intent(final_query)
        st.toast(f"Detected Intent: {intent}", icon="🤖")

        doc_chain, retriever = get_rag_components()
        
        if doc_chain:
            # B. Initial Retrieval
            initial_docs = retriever.invoke(final_query)
            
            # C. Smart Filtering based on LLM Intent
            candidate_docs = []
            for d in initial_docs:
                page_num = d.metadata.get("page", 0)
                # If factual, filter out TOC pages (0-3) and TOC dots
                if intent == "FACTUAL":
                    if page_num < 4 or d.page_content.count("....") > 5:
                        continue 
                candidate_docs.append(d)

            if not candidate_docs: 
                candidate_docs = initial_docs[:10]

            # D. Reranking
            doc_texts = [d.page_content for d in candidate_docs]
            rerank_results = vo.rerank(query=final_query, documents=doc_texts, model="rerank-2.5", top_k=4)
            final_docs = [candidate_docs[r.index] for r in rerank_results.results]
            
            # E. Generation
            response = doc_chain.invoke({"input": final_query, "chat_history": history_str, "context": final_docs})
            answer_text = response if isinstance(response, str) else str(response)

            st.session_state.memory.append((final_query, answer_text))
            
            # F. Audio & UI
            spoken_text = clean_text_for_speech(answer_text)
            audio_folder = r"C:\Users\junjo\Desktop\PRO\RAG\Budgest-AI\audio_responses"
            os.makedirs(audio_folder, exist_ok=True)
            tts_path = os.path.join(audio_folder, f"response_{uuid.uuid4().hex}.mp3")
            try:
                gTTS(text=spoken_text, lang='en').save(tts_path)
            except: tts_path = None

            with st.chat_message("assistant"):
                st.markdown(answer_text)
                if tts_path: st.audio(tts_path)
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(final_docs):
                        st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 0)+1}):**")
                        st.caption(doc.page_content)
                        st.divider()
            
            st.session_state.messages.append({"role": "assistant", "content": answer_text, "audio": tts_path, "sources": final_docs})
            st.rerun()