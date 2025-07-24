from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.pdf_processor import process_pdf
import traceback

router = APIRouter()

def process_pdf_and_return_stats(contents: bytes, filename: str):
    import fitz
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import time
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import onnxruntime as ort

    result = {
        "filename": filename,
        "extraction_time": None,
        "chunking_time": None,
        "embedding_time": None,
        "total_time": None,
        "num_chunks": 0,
        "num_embeddings": 0,
        "embedding_dim": None,
        "error": None
    }
    overall_start_time = time.perf_counter()
    # Extraction
    extraction_start_time = time.perf_counter()
    extracted_text = ""
    try:
        with fitz.open(stream=contents, filetype="pdf") as doc:
            for page in doc:
                extracted_text += page.get_text()
    except Exception as e:
        result["error"] = f"Error extracting text: {e}"
        result["traceback"] = traceback.format_exc()
        return result
    extraction_end_time = time.perf_counter()
    result["extraction_time"] = extraction_end_time - extraction_start_time
    if not extracted_text:
        result["error"] = "No text extracted from PDF."
        return result
    # Chunking
    chunking_start_time = time.perf_counter()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(extracted_text)
    chunking_end_time = time.perf_counter()
    result["chunking_time"] = chunking_end_time - chunking_start_time
    if not chunks:
        result["error"] = "No chunks generated from PDF."
        return result
    # Embedding
    embedding_start_time = time.perf_counter()
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    onnx_file_name = "onnx/model_qint8_avx512.onnx"
    try:
        embedder = SentenceTransformer(
            model_name_or_path=embed_model_name,
            cache_folder='./onnx_cache',
            backend="onnx",
            model_kwargs={"file_name": onnx_file_name},
        )
        embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    except Exception as e:
        result["error"] = f"Error during embedding: {e}"
        result["traceback"] = traceback.format_exc()
        return result
    embedding_end_time = time.perf_counter()
    result["embedding_time"] = embedding_end_time - embedding_start_time
    result["num_chunks"] = len(chunks)
    result["num_embeddings"] = len(embeddings)
    if len(embeddings) > 0:
        result["embedding_dim"] = len(embeddings[0])
    result["total_time"] = time.perf_counter() - overall_start_time
    return result

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    try:
        contents = await file.read()
        stats = process_pdf_and_return_stats(contents, file.filename)
        if stats.get("error"):
            return JSONResponse(content={"status": "error", **stats}, status_code=500)
        return JSONResponse(content={"status": "success", **stats}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e), "traceback": traceback.format_exc()}, status_code=500)
