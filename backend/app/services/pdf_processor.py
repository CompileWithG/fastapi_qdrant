import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import onnxruntime as ort

def process_pdf(contents: bytes, filename: str):
    print(f"Processing PDF: {filename}, size: {len(contents)} bytes")
    overall_start_time = time.perf_counter()

    # --- Step 1: Text Extraction ---
    extraction_start_time = time.perf_counter()
    extracted_text = ""
    try:
        with fitz.open(stream=contents, filetype="pdf") as doc:
            for page in doc:
                extracted_text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return
    extraction_end_time = time.perf_counter()
    extraction_time = extraction_end_time - extraction_start_time

    if not extracted_text:
        print(f"No text extracted from {filename}.")
        return

    # --- Step 2: Text Chunking ---
    chunking_start_time = time.perf_counter()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(extracted_text)
    chunking_end_time = time.perf_counter()
    chunking_time = chunking_end_time - chunking_start_time

    if not chunks:
        print(f"No chunks generated from {filename}.")
        return

    # --- Step 3: ONNX Quantized Embedding ---
    embedding_start_time = time.perf_counter()
    available_providers = []
    try:
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            print("CUDAExecutionProvider is available (GPU detected).")
        else:
            print("CUDAExecutionProvider not available. Using CPUExecutionProvider.")
    except ImportError:
        print("ONNX Runtime not fully installed, cannot check providers.")

    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    onnx_file_name = "onnx/model_qint8_avx512.onnx"  # You may want to make this configurable
    try:
        embedder = SentenceTransformer(
            model_name_or_path=embed_model_name,
            cache_folder='./onnx_cache',
            backend="onnx",
            model_kwargs={"file_name": onnx_file_name},
        )
        if "qint8" in onnx_file_name or "quantized" in onnx_file_name:
            print(f"Attempting to load quantized ONNX model: {onnx_file_name}")
        else:
            print(f"Loading standard ONNX model: {onnx_file_name}. Confirm it is truly quantized if expecting speedup.")
        embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    except Exception as e:
        print(f"Error loading or running ONNX embedding model: {e}")
        print(f"Attempted to load '{onnx_file_name}'.")
        print("Ensure 'onnx', 'onnxruntime', 'transformers', 'optimum', and 'sentence-transformers' are installed correctly.")
        print("Also, check internet connection for initial model download and file path/name correctness.")
        return
    embedding_end_time = time.perf_counter()
    embedding_time = embedding_end_time - embedding_start_time

    overall_end_time = time.perf_counter()
    total_time = overall_end_time - overall_start_time

    print(f"Time Breakdown: Extraction: {extraction_time:.4f}s, Chunking: {chunking_time:.4f}s, Embedding: {embedding_time:.4f}s")
    print(f"Successfully extracted, chunked, and embedded text from {filename}.")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings generated: {len(embeddings)}")
    if len(embeddings) > 0:
        print(f"Dimension of each embedding: {len(embeddings[0])}")
    print(f"Total end-to-end time taken: {total_time:.4f} seconds")
    print("\nFirst 3 chunks and their embedding previews:")
    for i in range(min(3, len(chunks))):
        print(f"--- Chunk {i+1} ---")
        print(chunks[i])
        print(f"Embedding Preview (first 5 values): {embeddings[i][:5]}...")
        print("-" * 20)
