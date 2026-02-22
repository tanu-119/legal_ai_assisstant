import chromadb
from chromadb.config import Settings

# --- 0. STREAMLIT CLOUD SQLITE FIX (MUST BE AT THE VERY TOP) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import json
import os
import glob

# Official v0.3+ Import Paths
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# --- 1. SETTINGS & INITIALIZATION ---
# Access the key securely from Streamlit Cloud Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
st.set_page_config(page_title="Legal AI Assistant", layout="wide")

# --- 2. DATA PROCESSING ---
def load_and_process_data():
    documents = []
    
    # Process ALL split judgement files (judgements_part_1.json, etc.)
    judgement_files = glob.glob("judgements_part_*.json")
    
    if not judgement_files:
        st.warning("No split judgement files found. Please ensure you ran the split script.")
    
    for filename in judgement_files:
        with open(filename, 'r', encoding='utf-8') as f:
            judgements = json.load(f)
            for j in judgements:
                # Combine fields into a single searchable text
                text = f"Title: {j.get('title', 'Unknown')}\n" \
                       f"Act: {j.get('act', 'N/A')}\n" \
                       f"Judge: {j.get('judge', 'N/A')}\n" \
                       f"Headnote: {' '.join(j.get('headnote_sent', []))}"
                
                documents.append(Document(
                    page_content=text, 
                    metadata={"source": filename, "id": j.get('case_id', 'N/A')}
                ))
    
    # Process IPC Sections
    if os.path.exists('ipc_sections.json'):
        with open('ipc_sections.json', 'r', encoding='utf-8') as f:
            ipc = json.load(f)
            if isinstance(ipc, dict): ipc = [ipc]
            for item in ipc:
                text = f"Section {item.get('Section')}: {item.get('section_title')}\n" \
                       f"Description: {item.get('section_desc')}"
                documents.append(Document(
                    page_content=text, 
                    metadata={"source": "ipc_sections.json"}
                ))
                
    return documents

# --- 3. BUILD THE BOT ENGINE ---
@st.cache_resource 
def setup_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "./legal_db_index"
    
    # 1. Use the most stable initialization for Streamlit Cloud
    # We remove 'is_persistent' and 'allow_reset' to avoid Pydantic errors
    client = chromadb.PersistentClient(path=persist_directory)

    try:
        # Check if collection exists
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        if "legal_collection" in collection_names:
            vectorstore = Chroma(
                client=client,
                collection_name="legal_collection",
                embedding_function=embeddings,
            )
        else:
            vectorstore = None
    except Exception:
        vectorstore = None

    # 2. Rebuild if vectorstore is None
    if vectorstore is None:
        docs = load_and_process_data()
        if not docs:
            st.error("No documents found to index!")
            st.stop()
            
        with st.status("Initializing Legal Database...", expanded=True) as status:
            vectorstore = Chroma.from_documents(
                documents=docs[:10], 
                embedding=embeddings,
                client=client,
                collection_name="legal_collection"
            )
            
            # Batch upload the remaining documents
            batch_size = 500 
            for i in range(10, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                vectorstore.add_documents(batch)
            status.update(label="Indexing Complete!", state="complete")
    
    # --- LLM and Chain Configuration remains the same ---
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.1-8b-instant", 
        temperature=0.1
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
    )

    custom_template = """You are a helpful Indian Legal Assistant. 
    Use the following pieces of retrieved context to answer the user's question.
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Helpful Legal Response:"""
    
    CUSTOM_PROMPT = PromptTemplate(
        template=custom_template, 
        input_variables=["context", "chat_history", "question"]
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=True
    )
# --- 4. STREAMLIT UI ---
st.title("⚖️ Indian Legal Assistant")
st.markdown("Providing legal clarity for the common man using IPC and Case Law data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the Bot
qa_bot = setup_qa_chain()

# Sidebar for controls
with st.sidebar:
    st.info("Ask about IPC sections or search for previous court judgements.")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal records..."):
            try:
                response = qa_bot({"question": prompt})
                answer = response['answer']
                st.markdown(answer)
                
                with st.expander("Show Legal Sources Used"):
                    for doc in response['source_documents']:
                        source_name = doc.metadata.get('source', 'Unknown')
                        st.info(f"**Source:** {source_name}\n\n{doc.page_content[:400]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")