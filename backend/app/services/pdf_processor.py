# app/services/pdf_processor.py
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
from openai import OpenAI
import json

class PDFProcessor:
    def __init__(self):
        self.document_embeddings = None
        self.question_embeddings = None
        self.chunks = None
        self.embedder = None
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "document_chunks"
        self.retrieved_answers = []
        self.deepseek_client = OpenAI(
            api_key="sk-90e085c5767b489faa59bdbac93540de",
            base_url="https://api.deepseek.com"
        )
        
        self._initialize_embedder()

    def _initialize_embedder(self):
        """Initialize the ONNX embedding model"""
        try:
            self.embedder = SentenceTransformer(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder='./onnx_cache',
                backend="onnx",
                model_kwargs={"file_name": "onnx/model_qint8_avx512.onnx"},
            )
        except Exception as e:
            print(f"Error initializing embedder: {e}")
            raise

    def _initialize_qdrant_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                ),
            )
        except Exception as e:
            print(f"Collection initialization note: {str(e)}")

    def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Failed to download PDF: {str(e)}")
            raise

    def extract_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            text = ""
            with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=64,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return text_splitter.split_text(text)

    def store_embeddings_in_qdrant(self):
        """Store document chunks and embeddings in Qdrant"""
        if not self.chunks or not self.document_embeddings:
            raise ValueError("No document chunks or embeddings to store")
        
        self._initialize_qdrant_collection()
        
        points = [
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"text": chunk}
            )
            for idx, (chunk, embedding) in enumerate(zip(self.chunks, self.document_embeddings))
        ]
        
        operation_info = self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        print(f"Stored {len(points)} document chunks in Qdrant")
        return operation_info

    def search_questions(self, questions: List[str]) -> None:
        """Search each question against Qdrant and store relevant text chunks"""
        if not questions or not self.question_embeddings:
            return
        
        self.retrieved_answers = []
        
        for question, embedding in zip(questions, self.question_embeddings):
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=3,
                with_payload=True,
                with_vectors=False
            )
            
            relevant_chunks = [result.payload['text'] for result in search_results]
            self.retrieved_answers.append({
                'question': question,
                'context': relevant_chunks
            })
        
        print(f"Retrieved context for {len(questions)} questions")

    def refine_with_deepseek(self) -> Dict:
        """Use DeepSeek to generate precise answers from retrieved context"""
        if not self.retrieved_answers:
            return {"answers": []}
        
        final_answers = []
        
        for qa_pair in self.retrieved_answers:
            question = qa_pair['question']
            context = "\n\n".join(qa_pair['context'])
            
            prompt = f"""
            **Document Analysis Task**
            
            **Question:** {question}
            
            **Relevant Context from Document:**
            {context}
            
            **Instructions:**
            1. Analyze the provided context thoroughly
            2. Identify all relevant information that answers the question
            3. Provide a comprehensive answer that:
               - Directly addresses the question
               - Includes supporting evidence from the context
               - Explains the reasoning behind your conclusion
               - Maintains accuracy to the original document
            4. If the context doesn't contain the answer, state "The document does not specify."
            
            **Required Format:**
            - Start with a clear, direct answer
            - Follow with reasoning/justification (marked with "Reasoning:")
            - Cite relevant parts of the context (marked with "Reference:")
            
            **Example Response:**
            "The document specifies a 30-day grace period for payments. 
            Reasoning: This allows flexibility while maintaining policy continuity. 
            Reference: Page 12 states 'All policies have a 30-day grace period...'"
            """
            
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert document analyst that provides thorough, evidence-based answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
                stream=False
            )
            
            answer = response.choices[0].message.content
            final_answers.append(answer)
        
        return {"answers": final_answers}

    def process_document(self, document_url: str):
        """Process document through the full pipeline"""
        try:
            pdf_content = self.download_pdf(document_url)
            text = self.extract_text(pdf_content)
            
            self.chunks = self.chunk_text(text)
            if not self.chunks:
                raise ValueError("No chunks generated from document")
            
            self.document_embeddings = self.embedder.encode(
                self.chunks,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            print(f"Generated {len(self.document_embeddings)} document embeddings")
            
            self.store_embeddings_in_qdrant()
            
        except Exception as e:
            print(f"Document processing failed: {e}")
            raise

    def process_questions(self, questions: List[str]):
        """Process list of questions to embeddings"""
        if not questions:
            return
            
        try:
            self.question_embeddings = self.embedder.encode(
                questions,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            print(f"Generated {len(self.question_embeddings)} question embeddings")
        except Exception as e:
            print(f"Question processing failed: {e}")
            raise

    def process(self, document_url: str, questions: List[str]) -> Dict:
        """
        Main processing method
        Returns: Dictionary with "answers" key containing refined responses
        """
        self.process_document(document_url)
        self.process_questions(questions)
        self.search_questions(questions)
        
        return self.refine_with_deepseek()
