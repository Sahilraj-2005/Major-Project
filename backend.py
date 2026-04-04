import os
import shutil
import time
import pandas as pd
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

IMAGE_OUTPUT_DIR = "extracted_data"

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
            
            text = "\n".join([line for line in result]) if result else "No text found in image."
            content = f"[IMAGE: {saved_img_path}]\nExtracted OCR Text:\n{text}"
            return [Document(page_content=content, metadata={"source": self.original_filename})]
        except Exception as e:
            print(f"OCR Error for {self.file_path}: {e}")
            return []

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
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in .env file")
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY not found in .env file")

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "mining-law")
        
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        
        # HuggingFace "all-MiniLM-L6-v2" outputs exactly 384 dimensions. This is required for the index.
        self.embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2
        )
        
        # Initialize Pinecone Client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Create the index automatically if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index '{self.index_name}' in the cloud. This takes ~30 seconds...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for the index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
                
        self.index = self.pc.Index(self.index_name)
        
        # Connect LangChain to the Pinecone index
        self.vector_store = PineconeVectorStore(
            index=self.index, 
            embedding=self.embedding_func,
            text_key="text"
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        self.prompt_template = PromptTemplate(
            template="""
            You are a highly capable AI assistant specializing in Mining Engineering and Legislation (Mines Act 1952, CMR 2017, MMR 1961, DGMS circulars).

            Your primary job is to extract strict factual data from the provided context and then explain it educationally. 

            TABLES & IMAGES:
            - The context might contain data from tables or references to images formatted strictly as [IMAGE: filepath].
            - If a table is relevant, include its data in a clean Markdown table.
            - If an image is relevant, you MUST output the exact tag [IMAGE: filepath] in the Regulation section.
            - IMPORTANT: If no relevant image tag is present in the context, do NOT mention images at all. Just answer using the text.

            If the answer is not in the context, say "I don't know based on the provided documents."

            Chat History:
            {chat_history}

            Context:
            {context}

            Question: {question}

            --- CRITICAL OUTPUT FORMATTING RULES ---
            You MUST structure your ENTIRE response exactly like the template below. 
            Do NOT add introductory sentences. 
            Do NOT merge the sections. 
            You MUST include the exact string '---EXPLANATION---' on its own line.

            [Provide the exact rules, clauses, and legislation here based on the context. Be strict, direct, and factual. Do not explain anything.]
            
            ---EXPLANATION---
            
            [Provide a thorough, easy-to-understand explanation of what those regulations mean in practical mining scenarios. Provide examples if helpful.]
            """,
            input_variables=["context", "question", "chat_history"]
        )

    def is_db_populated(self):
        """Helper method for Streamlit to check if the cloud database has data."""
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count > 0
        except Exception:
            return False

    def clear_database(self):
        # Delete all vectors from the Pinecone index
        try: 
            self.index.delete(delete_all=True)
        except Exception as e: 
            print(f"Error clearing Pinecone: {e}")
            
        import gc
        gc.collect()
        
        if os.path.exists(IMAGE_OUTPUT_DIR):
            try: shutil.rmtree(IMAGE_OUTPUT_DIR)
            except Exception: pass

    def process_pdf_with_images(self, file_path, original_filename):
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
        documents = []
        
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text("text") 
                except Exception as e:
                    print(f"⚠️ Skipping text on {original_filename} (Page {page_num}): {e}")
                    text = ""
                
                try:
                    image_list = page.get_images(full=True)
                except Exception as e:
                    print(f"⚠️ Skipping images on {original_filename} (Page {page_num}): {e}")
                    image_list = []
                
                for img_index, img in enumerate(image_list):
                    try:
                        if isinstance(img, tuple) and len(img) > 0:
                            xref = img
                        else:
                            xref = img
                            
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue
                            
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_filename = f"{original_filename}_p{page_num}_i{img_index}.{image_ext}"
                        image_filepath = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
                        
                        with open(image_filepath, "wb") as f:
                            f.write(image_bytes)
                        
                        ocr_text = ""
                        try:
                            from rapidocr_onnxruntime import RapidOCR
                            ocr = RapidOCR()
                            result, _ = ocr(image_filepath)
                            if result:
                                ocr_text = "\n".join([str(line) for line in result])
                        except Exception: 
                            pass 
                        
                        img_context = f"[IMAGE: {image_filepath}]\nImage Context (OCR): {ocr_text}\nSurrounding Page Text: {text[:300]}"
                        documents.append(Document(page_content=img_context, metadata={"source": original_filename, "type": "image"}))
                        
                    except Exception as e:
                        print(f"⚠️ Skipping specific image {img_index} on {original_filename} (Page {page_num}): {e}")
                
                if text and text.strip():
                    documents.append(Document(page_content=text, metadata={"source": original_filename, "type": "text"}))
                    
        except Exception as e:
            print(f"❌ Fatal error opening PDF {original_filename}: {e}")
            
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
                        docs = self.process_pdf_with_images(file_path, file.name)
                        documents.extend(docs)
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
                except Exception as e: print(f"Error processing {file.name}: {e}")

        if not documents: return 0

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
        if not chunks: return 0

        # Push the new chunks up to the Pinecone Cloud
        self.vector_store.add_documents(documents=chunks)
        
        return len(chunks)

    def retrieve_docs(self, query):
        if not self.retriever: return []
        return self.retriever.invoke(query)

    def get_answer_stream(self, query, context_docs, chat_history=[]):
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
            "chat_history": history_str
        })