import streamlit as st
import os
import re
from backend import RAGBackend

st.set_page_config(page_title="Mining Law Assistant", layout="wide")

st.title("⛏️ Comprehensive Mining Assistant")
st.markdown("Supports **CMR 2017, MMR 1961, Mines Act 1952, & DGMS Guidelines**.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend" not in st.session_state:
    try:
        with st.spinner("Connecting to Pinecone Cloud Database..."):
            st.session_state.backend = RAGBackend()
    except Exception as e:
        st.error(f"Startup Error: {e}. Check your .env file.")
        st.session_state.backend = None

with st.sidebar:
    st.header("1. System Status")
    
    # --- Check Pinecone Cloud Stats ---
    db_exists = False
    if st.session_state.backend:
        db_exists = st.session_state.backend.is_db_populated()
    
    if db_exists:
        st.success("✅ Cloud Database Pre-loaded & Ready!")
        st.caption("You can start asking questions immediately. The system is connected to Pinecone.")
        
        with st.expander("➕ Add new documents to the cloud database"):
            uploaded_files = st.file_uploader(
                "Upload Legislation/Books", 
                type=["pdf", "txt", "csv", "xlsx", "xls", "png", "jpg", "jpeg"], 
                accept_multiple_files=True
            )
            process_btn = st.button("Submit New Files")
    else:
        st.warning("⚠️ No cloud data found.")
        st.caption("Please upload your foundational mining legislation documents to populate Pinecone.")
        uploaded_files = st.file_uploader(
            "Upload Legislation/Books", 
            type=["pdf", "txt", "csv", "xlsx", "xls", "png", "jpg", "jpeg"], 
            accept_multiple_files=True
        )
        process_btn = st.button("Submit First Files")

    st.divider()
    if st.button("⚠️ Wipe Cloud Database", type="primary"):
        try:
            if st.session_state.backend:
                with st.spinner("Deleting all vectors from Pinecone..."):
                    st.session_state.backend.clear_database()
            st.session_state.messages = [] 
            st.session_state.backend = RAGBackend() 
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting: {e}")

    st.info("API Keys loaded from .env file")

# --- Processing Logic ---
if 'process_btn' in locals() and process_btn:
    if not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        try:
            with st.spinner("Processing and uploading new documents to Pinecone..."):
                if st.session_state.backend is None:
                    st.session_state.backend = RAGBackend()
                num_chunks = st.session_state.backend.process_documents(uploaded_files)
                if num_chunks > 0:
                    st.success(f"Successfully added {len(uploaded_files)} files into the cloud database!")
                    st.rerun() 
                else:
                    st.warning("Processed files but found no valid text chunks.")
        except Exception as e:
            st.error(f"Processing Error: {str(e)}")

st.divider()
st.header("2. Query Interface")

def render_message(role, full_content, images=None, sources=None):
    with st.chat_message(role):
        # Safely split the text using regex, catching both the strict delimiter and the AI's fallback
        split_pattern = r"---EXPLANATION---|(?:\*\*|)The Explanation \(Educational\):(?:\*\*|)"
        parts = re.split(split_pattern, full_content, maxsplit=1)
        
        if len(parts) > 1:
            reg_part, exp_part = parts
            
            st.markdown(reg_part.strip())
            with st.expander("📖 Learn More (Explanation)"):
                st.markdown(exp_part.strip())
        else:
            st.markdown(full_content)
        
        if images:
            for img_path in images:
                if os.path.exists(img_path):
                    st.image(img_path, caption="Sourced Image")

        if sources:
            with st.expander("View Sources & Metadata"):
                for src in sources:
                    st.markdown(f"**Source:** `{src['source']}` | **Chunk:** `{src['content'][:50]}...`")

for message in st.session_state.messages:
    render_message(message["role"], message["content"], message.get("images"), message.get("sources"))

if prompt := st.chat_input("E.g., What are the regulations for slope stability?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ensure the cloud database actually has data before querying
    if st.session_state.backend and st.session_state.backend.is_db_populated():
        with st.chat_message("assistant"):
            context_docs = st.session_state.backend.retrieve_docs(prompt)
            stream_generator = st.session_state.backend.get_answer_stream(
                query=prompt, 
                context_docs=context_docs, 
                chat_history=st.session_state.messages
            )
            
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream_generator:
                full_response += chunk
                
                # Dynamically hide the delimiter while it streams
                display_text = re.sub(
                    r"(---EXPLANATION---|(?:\*\*|)The Explanation \(Educational\):(?:\*\*|))", 
                    "\n\n**Preparing Explanation...**\n\n", 
                    full_response
                )
                message_placeholder.markdown(display_text + "▌")
            
            message_placeholder.empty()
            
            images_to_show = re.findall(r"\[IMAGE:\s*(.*?)\]", full_response)
            valid_images = [p for p in images_to_show if os.path.exists(p)]
            
            sources_data = [{"source": d.metadata.get("source", "Unknown"), "content": d.page_content.replace("\n", " ")} for d in context_docs]

        render_message("assistant", full_response, valid_images, sources_data)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "images": valid_images,
            "sources": sources_data
        })
    else:
        warning_msg = "Database is empty. Please upload your documents first to populate Pinecone."
        with st.chat_message("assistant"):
            st.warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})