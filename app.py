__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

import json
import os

# Official v0.3+ Import Paths
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# --- 1. SETTINGS & INITIALIZATION ---
# NOTE: Please generate a new key on Groq and keep it private!
GROQ_API_KEY = "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" 
st.set_page_config(page_title="Legal AI Assistant", layout="wide")

# --- 2. DATA PROCESSING ---
def load_and_process_data():
    documents = []
    
    if os.path.exists('judgements.json'):
        with open('judgements.json', 'r', encoding='utf-8') as f:
            judgements = json.load(f)
            for j in judgements:
                text = f"Title: {j['title']}\nAct: {j['act']}\nJudge: {j['judge']}\nHeadnote: {' '.join(j['headnote_sent'])}"
                documents.append(Document(page_content=text, metadata={"source": "judgements.json", "id": j.get('case_id', 'N/A')}))
    
    if os.path.exists('ipc_sections.json'):
        with open('ipc_sections.json', 'r', encoding='utf-8') as f:
            ipc = json.load(f)
            if isinstance(ipc, dict): ipc = [ipc]
            for item in ipc:
                text = f"Section {item.get('Section')}: {item.get('section_title')}\nDescription: {item.get('section_desc')}"
                documents.append(Document(page_content=text, metadata={"source": "ipc_sections.json"}))
                
    return documents

# --- 3. BUILD THE BOT ENGINE ---
@st.cache_resource 
def setup_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "./legal_db_index"
    
    # 1. Initialize the client simply. 
    # Do not pass 'Settings' objects here to avoid Pydantic validation errors.
    import chromadb
    client = chromadb.PersistentClient(path=persist_directory)

    # 2. Use the LangChain wrapper with the existing client
    # We use 'legal_collection' to ensure consistency
    vectorstore = Chroma(
        client=client,
        collection_name="legal_collection",
        embedding_function=embeddings,
    )

    # 3. Check if we need to add data
    # client.list_collections() returns a list of Collection objects
    collections = client.list_collections()
    is_empty = True
    for col in collections:
        if col.name == "legal_collection" and col.count() > 0:
            is_empty = False
            break

    if is_empty:
        docs = load_and_process_data()
        if not docs:
            st.error("No documents found to index!")
            st.stop()
            
        with st.status("Indexing legal records...", expanded=True) as status:
            batch_size = 500 
            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                vectorstore.add_documents(batch)
            status.update(label="Indexing Complete!", state="complete")
    
    # --- LLM and Chain Logic ---
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

try:
    qa_bot = setup_qa_chain()
except Exception as e:
    st.error(f"Error initializing: {e}")
    st.stop()

# Sidebar for controls
with st.sidebar:
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal records..."):
            response = qa_bot({"question": prompt})
            answer = response['answer']
            st.markdown(answer)
            
            with st.expander("Show Legal Sources Used"):
                for doc in response['source_documents']:
                    st.info(f"Source: {doc.metadata['source']}\n\n{doc.page_content[:400]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})