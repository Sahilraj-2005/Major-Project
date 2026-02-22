import os
import shutil
import pandas as pd
import fitz  # PyMuPDF for handling PDFs and extracting images
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# Set up local directory to store extracted images so UI can access them
IMAGE_OUTPUT_DIR = "extracted_data"

# --- Custom OCR Loader (Updated to save image for rendering) ---
class CustomRapidOCRLoader:
    def __init__(self, file_path, original_filename):
        self.file_path = file_path
        self.original_filename = original_filename

    def load(self):
        try:
            os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
            saved_img_path = os.path.join(IMAGE_OUTPUT_DIR, self.original_filename)
            shutil.copy(self.file_path, saved_img_path)

            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
            result, _ = ocr(self.file_path)
            
            text = "\n".join([line[1] for line in result]) if result else "No text found in image."
            
            # Embed image path so LLM can return it if relevant
            content = f"[IMAGE: {saved_img_path}]\nExtracted OCR Text:\n{text}"
            return [Document(page_content=content, metadata={"source": self.original_filename})]
        except Exception as e:
            print(f"OCR Error for {self.file_path}: {e}")
            return []

# --- Custom Excel Loader (Unchanged) ---
class CustomPandasExcelLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        try:
            dfs = pd.read_excel(self.file_path, sheet_name=None)
            documents = []
            for sheet_name, df in dfs.items():
                for _, row in df.iterrows():
                    content = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    if content.strip():
                        documents.append(Document(
                            page_content=content, 
                            metadata={"source": self.file_path, "sheet": sheet_name}
                        ))
            return documents
        except Exception as e:
            print(f"Excel Error for {self.file_path}: {e}")
            return []

class RAGBackend:
    def __init__(self):
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.vector_store = None
        self.retriever = None
        
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        
        self.embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            temperature=0.2
        )
        
        # Updated Prompt: Handing Expertise Levels and Image/Table Retrieval
        self.prompt_template = PromptTemplate(
            template="""
            You are a helpful assistant for a Mini RAG Evaluation Project.
            You are speaking to a: {expertise_level}.

            BEHAVIOR RULES:
            - If talking to an "Expert": Provide EXACT rules and regulations related to their query. Provide NO explanations or fluff. Be strict, direct, and factual.
            - If talking to a "Normal Person": Explain the rules and regulations thoroughly and in simple, accessible terms. Provide examples if helpful.

            TABLES & IMAGES:
            - The context might contain data from tables or references to images formatted strictly as [IMAGE: filepath].
            - If an image/table is relevant to the question, you MUST include its data. Provide table data in a clean Markdown table.
            - If an image is relevant, you MUST output the exact tag [IMAGE: filepath] in your response so the system can display it. Provide context around the image based on the OCR data.

            If the answer is not in the context, say "I don't know based on the provided documents."

            Chat History:
            {chat_history}

            Context:
            {context}

            Question: {question}
            """,
            input_variables=["context", "question", "chat_history", "expertise_level"]
        )

        if os.path.exists("./chroma_db"):
            print("Found existing vector store. Loading...")
            self.vector_store = Chroma(
                persist_directory="./chroma_db", 
                embedding_function=self.embedding_func
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

    def clear_database(self):
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except Exception: pass

        self.vector_store = None
        self.retriever = None
        
        import gc
        gc.collect()

        if os.path.exists("./chroma_db"):
            try: shutil.rmtree("./chroma_db")
            except Exception: pass

        if os.path.exists(IMAGE_OUTPUT_DIR):
            try: shutil.rmtree(IMAGE_OUTPUT_DIR)
            except Exception: pass

    # --- New: PDF Processor with Image & Table extraction ---
    def process_pdf_with_images(self, file_path, original_filename):
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
        doc = fitz.open(file_path)
        documents = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text") 
            
            # Extract images from PDF
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{original_filename}_p{page_num}_i{img_index}.{image_ext}"
                image_filepath = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
                
                with open(image_filepath, "wb") as f:
                    f.write(image_bytes)
                
                # Perform OCR on extracted image for context
                ocr_text = ""
                try:
                    from rapidocr_onnxruntime import RapidOCR
                    ocr = RapidOCR()
                    result, _ = ocr(image_filepath)
                    if result:
                        ocr_text = "\n".join([line[1] for line in result])
                except Exception: pass
                
                img_context = f"[IMAGE: {image_filepath}]\nImage Context (OCR): {ocr_text}\nSurrounding Page Text: {text[:300]}"
                documents.append(Document(page_content=img_context, metadata={"source": original_filename, "type": "image"}))
            
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": original_filename, "type": "text"}))
        
        return documents

    def process_documents(self, uploaded_files):
        import tempfile
        documents = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                try:
                    loader = None
                    if file.name.lower().endswith(".pdf"):
                        # Replaced PyPDFLoader with Custom PyMuPDF logic
                        docs = self.process_pdf_with_images(file_path, file.name)
                        documents.extend(docs)
                        print(f"Loaded {len(docs)} chunks from PDF {file.name}")
                        continue
                        
                    elif file.name.lower().endswith(".txt"):
                        loader = TextLoader(file_path)
                    elif file.name.lower().endswith(".csv"):
                        encodings = ['utf-8', 'cp1252', 'latin-1']
                        for enc in encodings:
                            try:
                                loader = CSVLoader(file_path=file_path, encoding=enc)
                                loader.load()
                                break 
                            except Exception: loader = None
                    elif file.name.lower().endswith((".xlsx", ".xls")):
                        loader = CustomPandasExcelLoader(file_path)
                    elif file.name.lower().endswith((".png", ".jpg", ".jpeg")):
                        loader = CustomRapidOCRLoader(file_path, file.name)
                    
                    if loader:
                        docs = loader.load()
                        valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
                        for doc in valid_docs:
                            if "source" not in doc.metadata:
                                doc.metadata["source"] = file.name
                        documents.extend(valid_docs)
                        
                except Exception as e:
                    print(f"Error processing {file.name}: {e}")

        if not documents:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            return 0

        self.vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embedding_func,
            persist_directory="./chroma_db"
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        return len(chunks)

    def retrieve_docs(self, query):
        if not self.retriever:
            return []
        return self.retriever.invoke(query)

    def get_answer_stream(self, query, context_docs, chat_history=[], expertise_level="Normal Person"):
        context_text = "\n\n".join(doc.page_content for doc in context_docs)
        
        history_str = ""
        for msg in chat_history:
            role = msg.get("role", "User")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"

        chain = self.prompt_template | self.llm | StrOutputParser()
        
        return chain.stream({
            "context": context_text, 
            "question": query, 
            "chat_history": history_str,
            "expertise_level": expertise_level
        })