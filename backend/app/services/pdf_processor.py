# app/services/pdf_processor.py
import fitz
from docx import Document
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from qdrant_client import QdrantClient
#from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import (
    VectorParams, Distance,
    OptimizersConfigDiff, HnswConfigDiff, WalConfigDiff,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,PointStruct
)

from qdrant_client.http import models  # For SearchParams and other models
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
        self.question=[]
        self.document_embeddings = None
        self.question_embeddings = None
        self.chunks = None
        self.final_answers = []
        self.embedder = None
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "document_chunks"
        self.retrieved_answers = []
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
         # Initialize tokenizer (add this to your __init__ method)
        if not hasattr(self, 'tokenizer'):
            from transformers import GPT2TokenizerFast
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        #print("---------------------------------------------------------------------------------------------------")
        self._initialize_embedder()
        self._initialize_qdrant_collection()

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
            #print("==================================================================================================================")
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
        distance=Distance.COSINE,
        on_disk=False # Reduces RAM usage without significant performance hit
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=0,  # Build index after 20k vectors
        memmap_threshold=0,   # Use memory-mapped files
        default_segment_number=8,
        max_optimization_threads=4  # Parallel processing
    ),
    hnsw_config=HnswConfigDiff(
        m=24,                     # Higher connectivity (16-32 is optimal)
        ef_construct=75,         # Higher quality index build
        full_scan_threshold=5000, # Use HNSW for collections >5k vectors
        on_disk=False            # Store graph on disk
    ),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8, # 8-bit quantization
            quantile=0.95,        # Preserve 95% accuracy
            always_ram=True       # Keep quantized vectors in RAM
        )
    ),
    wal_config=WalConfigDiff(
        wal_capacity_mb=1024,     # Larger WAL for bulk inserts
        wal_segments_ahead=2      # Better write throughput
    )
)
            self.print_elapsed_time("_initialize_qdrant_collection") 
        except Exception as e:
            print(f"Collection initialization note: {str(e)}")

    def _clear_qdrant_collection(self):
        """Delete all vectors from the Qdrant collection"""
        try:
            self.qdrant_client.delete(collection_name=self.collection_name,points_selector=models.FilterSelector(filter=models.Filter()  # Empty filter matches all points
                )
            )       
            print(f"Cleared vectors inside : {self.collection_name}")
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
                #pdf processing
                text = ""
                with fitz.open(stream=content, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text()
                #print(text)
                return text
            
            elif filetype in ['docx', 'doc']:
                #doc processing
                doc = Document(BytesIO(content))
                return '\n'.join([para.text for para in doc.paragraphs])
            
            elif filetype == 'eml':
                msg = BytesParser(policy=policy.default).parsebytes(content)
                text = ""
                #email processing
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
            chunk_size=768,
            chunk_overlap=0,
            separators=["\n\n", "\n", "."],
            length_function=lambda x: len(x.encode('utf-8')),
            keep_separator=False
        )
        self.print_elapsed_time("chunk_text")
        #print(text_splitter.split_text(text))
        return text_splitter.split_text(text)

    def store_embeddings_in_qdrant(self):
        """Store document chunks and embeddings in Qdrant"""
        if  self.document_embeddings is None:
            raise ValueError("No document chunks or embeddings to store")
        
        
        
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
            wait=False,  # Use False for bulk inserts to improve performance
        )
        print(f"Stored {len(points)} document chunks in Qdrant")
        self.print_elapsed_time("store_embeddings_in_qdrant")

    def search_questions(self, questions: List[str]) -> None:
        """Search each question against Qdrant and store relevant text chunks"""
        if  self.question_embeddings is None:
            return
        
        self.retrieved_answers = []
        
        for question, embedding in zip(questions, self.question_embeddings):
            search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=4,
            with_payload=True,
            with_vectors=False,  # Don't return full vectors to reduce payload
            search_params=models.SearchParams(
            hnsw_ef=32,  # Higher value = more accurate but slower (32-256 typical)
            exact=False   # Use approximate search (False for HNSW, True for brute-force)
            ),
            consistency="majority",  # Wait for at least 1 node to confirm write
            )
            relevant_chunks = []
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
        #print(f"Retrieved answers: {self.retrieved_answers}")

    def refine_with_deepseek(self) -> Dict:
        """Use DeepSeek to generate answers for all questions in one go"""
        if not self.retrieved_answers:
            return {"answers": []}

        

        # Calculate static prompt tokens (without the questions/context)
        static_prompt_template = """
            Document Analysis Task

            You will be given multiple questions and their relevant context from a document.

            {questions_contexts}

            Instructions(strictly must follow):
            For each question:
            1. Carefully analyze the provided context.
            2. Identify all key information needed to answer the question.
            3. Provide a clear and complete response that:
            - Directly answers the question
            - Explains why that answer is correct
            - Includes the relevant supporting evidence from the document as long as it doesnt go beyond the scope of whats asked in the question
            4. make sure the answers are in the same order as the questions,the answer number must match the question number
            5.strictly answer every question,the number of answers must match the number of questions.

            Output Format:
            - Output ONLY a plain Python list of strings.
            Be concise, accurate, and consistent,don't give extra details unless its relevant to the question asked. Only return the final list of responses.
            - Do NOT include markdown, code blocks, escape characters, or backticks.
            - Each item in the list must be a natural, complete sentence or paragraph that blends the answer, reasoning, and reference into a single string.
            - Do not use labels like "Answer:", "Reasoning:", or "Reference:".
            - (strictly must follow this)If a question doesnt have enough context to answer, return the given  string for that specific  question only  : "The document does not specify.",even if there is something remotly relevant to the question include it in the answer except of "document does not specify" string.
            - Your output must be structured like this:

        [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod odio vitae nisl ultricies, eget fermentum ipsum tempor. Proin auctor metus in libero.",
        "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Cras convallis tellus ac quam tincidunt (36) varius. Pellentesque habitant morbi tristique senectus.",
        "Curabitur, yes lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam (24) months. Quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Two (2) years sint occaecat cupidatat non proident."
        
    ]
            Strictly follow this format. Do not include any headings, JSON objects, markdown syntax, escape characters, or code fences.
            """
        
        static_tokens = len(self.tokenizer.encode(static_prompt_template.replace("{questions_contexts}", "")))
        max_available_tokens = 16385 - static_tokens - 500  # 500 token buffer for response

        questions_contexts = ""
        included_questions = 0
        current_tokens = 0

        for idx, qa_pair in enumerate(self.retrieved_answers, 1):
            question = qa_pair['question']
            context = "\n\n".join(qa_pair['context'])
            new_block = f"\n\n**Question {idx}:** {question}\n**Relevant Context:**\n{context}\n"
            new_block_tokens = len(self.tokenizer.encode(new_block))

            if current_tokens + new_block_tokens > max_available_tokens:
                break
                
            questions_contexts += new_block
            current_tokens += new_block_tokens
            included_questions += 1

        # Build the final prompt (unchanged from original)
        prompt = static_prompt_template.replace("{questions_contexts}", questions_contexts)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000  # Reserve tokens for response
            )
            
            self.print_elapsed_time("refine_with_llm")
            formatted_answer = response.choices[0].message.content
            self.final_answers = json.loads(formatted_answer)
            
            # Pad answers for excluded questions
            remaining_answers = len(self.retrieved_answers) - included_questions
            if remaining_answers > 0:
                self.final_answers.extend(["The document does not specify."] * remaining_answers)
                
        except Exception as e:
            print(f"Error generating answer with DeepSeek: {e}")
            self.final_answers = ["The document does not specify." for _ in self.retrieved_answers]

        return {"answers": self.final_answers}

    def process_document(self, document_url: str):
        """Process document through the full pipeline"""
        try:
            file_content, filetype = self.download_file(document_url)#bytes,str
            text = self.extract_text(file_content, filetype)#str
            
            self.chunks = self.chunk_text(text)
            if not self.chunks:
                raise ValueError("No chunks generated from document")
            
            self.document_embeddings = self.embedder.encode(
                self.chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=256,
                show_progress_bar=False,

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
            print(f"Question embeddings: {self.question_embeddings}")
            self.question = questions
        except Exception as e:
            print(f"Question processing failed: {e}")
            raise

    def process(self, document_url: str, questions: List[str]) -> Dict:
        """
        Main processing method
        Returns: Dictionary with "answers" key containing refined responses
        """
        try:
            if not document_url or not questions:
                return {"answers": ["Invalid input. Please provide a valid document URL and questions."]}
            self.process_document(document_url)
            self.process_questions(questions)
            self.search_questions(questions)
            self._clear_qdrant_collection()
            return self.refine_with_deepseek()
        except Exception as e:
            print(f"Processing failed: {e}")
            return {"answers": ["An error occurred during processing. Please try again later."]}