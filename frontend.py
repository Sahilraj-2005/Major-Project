import streamlit as st
import os
import re
from backend import RAGBackend

st.set_page_config(page_title="Mini RAG Assistant", layout="wide")

st.title("⚡ Mini RAG Assistant")
st.markdown("Supports: **PDF, TXT, CSV, Excel, Images (OCR)**. Powered by **Groq** & **HuggingFace**.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend" not in st.session_state:
    try:
        st.session_state.backend = RAGBackend()
    except Exception as e:
        st.error(f"Startup Error: {e}. Check your .env file.")
        st.session_state.backend = None

# --- Sidebar: Configuration & Ingestion ---
with st.sidebar:
    st.header("1. User Profile")
    # New: Expertise Selector
    expertise_level = st.radio(
        "Who is asking?", 
        ["Normal Person", "Expert"], 
        help="Experts receive exact rules/regulations with no explanation. Normal people receive detailed explanations."
    )
    
    st.divider()
    st.header("2. Document Corpus")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt", "csv", "xlsx", "xls", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        process_btn = st.button("Submit")
        
    with col2:
        reset_btn = st.button("⚠️ Reset", type="primary")

    if reset_btn:
        try:
            if st.session_state.backend:
                st.session_state.backend.clear_database()
            st.session_state.messages = [] 
            st.session_state.backend = RAGBackend() 
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting: {e}")

    st.info("API Keys loaded from .env file")

# --- Logic: Processing ---
if process_btn:
    if not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        try:
            with st.spinner("Processing documents (OCR, Chunking, Embedding, Image Extraction)..."):
                if st.session_state.backend is None:
                    st.session_state.backend = RAGBackend()
                
                backend = st.session_state.backend
                num_chunks = backend.process_documents(uploaded_files)
                
                if num_chunks > 0:
                    st.success(f"Successfully processed {len(uploaded_files)} files into {num_chunks} chunks!")
                else:
                    st.warning("Processed files but found no valid text chunks.")
        except Exception as e:
            st.error(f"Processing Error: {str(e)}")

# --- Main Interface: Chat ---
st.divider()
st.header("3. Query Interface")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display dynamically parsed images from previous responses
        if "images" in message and message["images"]:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, caption="Sourced Image")

        if "sources" in message:
            with st.expander("View Sources & Metadata"):
                for src in message["sources"]:
                    st.markdown(f"**Source:** `{src['source']}` | **Chunk:** `{src['content'][:50]}...`")

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.backend:
        with st.chat_message("assistant"):
            # 1. Retrieve Docs
            context_docs = st.session_state.backend.retrieve_docs(prompt)
            
            # 2. Stream Answer with Expertise Level
            stream_generator = st.session_state.backend.get_answer_stream(
                query=prompt, 
                context_docs=context_docs, 
                chat_history=st.session_state.messages,
                expertise_level=expertise_level
            )
            response = st.write_stream(stream_generator)
            
            # 3. Parse extracted images to show directly in UI
            images_to_show = re.findall(r"\[IMAGE:\s*(.*?)\]", response)
            valid_images = []
            for img_path in images_to_show:
                if os.path.exists(img_path):
                    st.image(img_path, caption="Relevant Extracted Image")
                    valid_images.append(img_path)

            # 4. Format Sources
            sources_data = []
            with st.expander("View Sources & Metadata"):
                for doc in context_docs:
                    source_name = doc.metadata.get("source", "Unknown")
                    page_content = doc.page_content.replace("\n", " ")
                    st.markdown(f"- **Source:** `{source_name}`\n  > {page_content[:100]}...")
                    sources_data.append({"source": source_name, "content": page_content})

        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "images": valid_images,
            "sources": sources_data
        })
    else:
        warning_msg = "Backend not initialized. Please upload and Submit documents."
        with st.chat_message("assistant"):
            st.warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})