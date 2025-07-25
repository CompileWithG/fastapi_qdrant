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
from dotenv import load_dotenv
load_dotenv()
import os
import time






class PDFProcessor:
    def print_elapsed_time(self, message=""):
        elapsed = time.time() - self.start_time
        print(f"[+{elapsed:.2f}s] {message}")


    def __init__(self):
        self.document_embeddings = None
        self.question_embeddings = None
        self.chunks = None
        self.final_answers = []
        self.embedder = None
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "document_chunks"
        self.retrieved_answers = []
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key = os.getenv("API_KEY")
        )
        
        self._initialize_embedder()

    def _initialize_embedder(self):
        self.start_time = time.time()
        """Initialize the ONNX embedding model"""
        try:
            self.embedder = SentenceTransformer(
                model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder='./onnx_cache',
                backend="onnx",
                model_kwargs={"file_name": "onnx/model_qint8_avx512.onnx"},
            )
            self.print_elapsed_time("_initialize_embedder") 
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
            self.print_elapsed_time("_initialize_qdrant_collection") 
        except Exception as e:
            print(f"Collection initialization note: {str(e)}")

    def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.print_elapsed_time()
            self.print_elapsed_time("download_pdf")

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
            self.print_elapsed_time("extract_text")
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
        self.print_elapsed_time("chunk_text")
        return text_splitter.split_text(text)

    def store_embeddings_in_qdrant(self):
        """Store document chunks and embeddings in Qdrant"""
        if self.chunks is None or self.document_embeddings is None:
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
        self.print_elapsed_time("store_embeddings_in_qdrant")
        return operation_info

    def search_questions(self, questions: List[str]) -> None:
        """Search each question against Qdrant and store relevant text chunks"""
        if not questions or self.question_embeddings is None:
            return
        
        self.retrieved_answers = []
        
        for question, embedding in zip(questions, self.question_embeddings):
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=embedding.tolist(),
                limit=3,
                with_payload=True,
            ).points
            #print(search_results)
            relevant_chunks = [
            result.payload['text']
            for result in search_results
            if result.payload and 'text' in result.payload
    ]

            #print("eroor occuers here")
            self.retrieved_answers.append({
                'question': question,
                'context': relevant_chunks
            })
        #print(self.retrieved_answers)
        print(f"Retrieved context for {len(questions)} questions")
        self.print_elapsed_time("search_questions")
    



#deepseek propmt fix

    def refine_with_deepseek(self) -> Dict:
        """Use DeepSeek to generate answers for all questions in one go"""
        if not self.retrieved_answers:
            return {"answers": []}

        # Build the prompt for all questions
        questions_contexts = ""
        for idx, qa_pair in enumerate(self.retrieved_answers, 1):
            question = qa_pair['question']
            context = "\n\n".join(qa_pair['context'])
            questions_contexts += f"\n\n**Question {idx}:** {question}\n**Relevant Context:**\n{context}\n"

        prompt = f"""
            Document Analysis Task

            You will be given multiple questions and their relevant context from a document.

            {questions_contexts}

            Instructions:
            For each question:
            1. Carefully analyze the provided context.
            2. Identify all key information needed to answer the question.
            3. Provide a clear and complete response that:
            - Directly answers the question
            - Explains why that answer is correct
            - Includes the relevant supporting evidence from the document

            Output Format:
            - Output ONLY a plain Python list of strings.
            - Do NOT include markdown, code blocks, escape characters, or backticks.
            - Each item in the list must be a natural, complete sentence or paragraph that blends the answer, reasoning, and reference into a single string.
            - Do not use labels like "Answer:", "Reasoning:", or "Reference:".
            - Your output must look exactly like this:

            [
            "A grace period of thirty days is provided after the due date for premium payment, which ensures continuity of coverage without loss of benefits, as clearly mentioned in the document: 'A grace period of 30 days is allowed for payment of premium.'",
            "The policy covers pre-existing diseases only after thirty-six months of continuous coverage, which helps the insurer manage initial high-risk liabilities, as stated in the clause: 'Pre-existing diseases will be covered after 36 months of continuous coverage.'"
            ]

            Strictly follow this format. Do not include any headings, JSON objects, markdown syntax, escape characters, or code fences.
            """


        try:
            completion = self.client.chat.completions.create(
                extra_body={},
                model="mistralai/mistral-small-3.2-24b-instruct:free",
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.print_elapsed_time("refine_with_llm")
            
            
            #final_ans fixed
            formatted_answer = completion.choices[0].message.content
            self.final_answers=formatted_answer
        except Exception as e:
            print(f"Error generating answer with DeepSeek: {e}")
            self.final_answers.append("The document does not specify.")

        return {"answers": self.final_answers}


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
