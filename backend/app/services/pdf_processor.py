# app/services/pdf_processor.py
import fitz
from docx import Document
from io import BytesIO
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
import urllib.parse
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse

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
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
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

    def _clear_qdrant_collection(self):
        """Delete all vectors from the Qdrant collection"""
        try:
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
            print(f"Cleared Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing Qdrant collection: {e}")

    def get_file_extension(self, url: str) -> str:
        """Extract file extension from URL (ignoring query parameters)"""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        return os.path.splitext(path)[1].lower()

    def download_file(self, url: str) -> tuple:
        """Download file and return (content, filetype)"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type:
                filetype = 'pdf'
            elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                filetype = 'docx'
            elif 'application/msword' in content_type:
                filetype = 'doc'
            elif 'message/rfc822' in content_type:
                filetype = 'eml'
            else:
                path = urlparse(url).path
                if path.endswith('.pdf'):
                    filetype = 'pdf'
                elif path.endswith('.docx'):
                    filetype = 'docx'
                elif path.endswith('.doc'):
                    filetype = 'doc'
                elif path.endswith('.eml'):
                    filetype = 'eml'
                else:
                    filetype = 'pdf'  # Default assumption
            
            self.print_elapsed_time("download_file")
            return response.content, filetype
        except Exception as e:
            print(f"Failed to download file: {str(e)}")
            raise

    def extract_text(self, content: bytes, filetype: str) -> str:
        """Extract text from bytes based on filetype"""
        try:
            if filetype == 'pdf':
                text = ""
                with fitz.open(stream=content, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text()
                return text
            
            elif filetype in ['docx', 'doc']:
                doc = Document(BytesIO(content))
                return '\n'.join([para.text for para in doc.paragraphs])
            
            elif filetype == 'eml':
                msg = BytesParser(policy=policy.default).parsebytes(content)
                text = ""
                
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        charset = part.get_content_charset() or 'utf-8'
                        if content_type == 'text/plain':
                            payload = part.get_payload(decode=True)
                            text += payload.decode(charset, errors='replace') + "\n\n"
                else:
                    payload = msg.get_payload(decode=True)
                    charset = msg.get_content_charset() or 'utf-8'
                    text = payload.decode(charset, errors='replace')
                
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
                limit=2,
                with_payload=True,
            ).points
            
            relevant_chunks = [
                result.payload['text']
                for result in search_results
                if result.payload and 'text' in result.payload
            ]

            self.retrieved_answers.append({
                'question': question,
                'context': relevant_chunks
            })
        
        print(f"Retrieved context for {len(questions)} questions")
        self.print_elapsed_time("search_questions")

    def refine_with_deepseek(self) -> Dict:
        """Use DeepSeek to generate answers for all questions in one go"""
        if not self.retrieved_answers:
            return {"answers": []}

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
            - If any question doesnt have enough context to answer, return a single string: "The document does not specify."
            - Your output must be structured like this:

           [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod odio vitae nisl ultricies, eget fermentum ipsum tempor. Proin auctor metus in libero.",
    "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Cras convallis tellus ac quam tincidunt (36) varius. Pellentesque habitant morbi tristique senectus.",
    "Curabitur, yes lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam (24) months. Quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Two (2) years sint occaecat cupidatat non proident.",
    "Yes, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.",
    "A No Claim Discount of 5% on the base lorem ipsum dolor sit amet. Consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Yes, the lorem ipsum policy reimburses expenses for health check-ups at the end of every block of two continuous policy years. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit.",
    "A hospital is defined as lorem ipsum dolor sit amet, consectetur (10) inpatient beds or (15) beds, with qualified nursing staff available 24/7. Neque porro quisquam est qui dolorem ipsum quia dolor sit amet.",
    "The policy covers medical expenses for inpatient treatment under Lorem, Ipsum, Dolor, Sit, Amet, and Consectetur systems. Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Lorem Ipsum, and ICU charges are capped at 2%. At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum."
]

            Strictly follow this format. Do not include any headings, JSON objects, markdown syntax, escape characters, or code fences.
            """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.print_elapsed_time("refine_with_llm")
            formatted_answer = response.choices[0].message.content
            self.final_answers = formatted_answer
        except Exception as e:
            print(f"Error generating answer with DeepSeek: {e}")
            self.final_answers = ["The document does not specify."]

        return {"answers": self.final_answers}

    def process_document(self, document_url: str):
        """Process document through the full pipeline"""
        try:
            file_content, filetype = self.download_file(document_url)
            text = self.extract_text(file_content, filetype)
            
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
        try:
            self.process_document(document_url)
            self.process_questions(questions)
            self.search_questions(questions)
            self._clear_qdrant_collection()
            return self.refine_with_deepseek()
        except Exception as e:
            print(f"Processing failed: {e}")
            return {"answers": ["An error occurred during processing. Please try again later."]}