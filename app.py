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
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
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
    
    import chromadb
    client = chromadb.PersistentClient(path=persist_directory)

    # 1. Check if the collection exists and has data
    collection_name = "legal_collection"
    collections = client.list_collections()
    exists_and_filled = any(col.name == collection_name and col.count() > 0 for col in collections)

    if not exists_and_filled:
        docs = load_and_process_data()
        if not docs:
            st.error("No documents found! Please check if judgements.json and ipc_sections.json exist.")
            st.stop()
            
        with st.status("First-time setup: Indexing legal records...", expanded=True) as status:
            # Create the vectorstore AND add documents in one go for the first time
            vectorstore = Chroma.from_documents(
                documents=docs[:100], # Start with a small batch to initialize
                embedding=embeddings,
                client=client,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            
            # Add the rest in batches
            batch_size = 500 
            for i in range(100, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                vectorstore.add_documents(batch)
            status.update(label="Indexing Complete!", state="complete")
    else:
        # Load the existing vectorstore
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

    # --- LLM and Chain Logic ---
    # Using st.secrets is highly recommended over hardcoding
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