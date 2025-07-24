from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.pdf_processor import process_pdf
import traceback
from pydantic import BaseModel
from typing import List
import httpx

router = APIRouter()

class HackrxRunRequest(BaseModel):
    documents: str 
    questions: List[str] = []

class HackrxRunResponse(BaseModel):
    status: str
    filename: str = None
    extraction_time: float = None
    chunking_time: float = None
    embedding_time: float = None
    total_time: float = None
    num_chunks: int = None
    num_embeddings: int = None
    embedding_dim: int = None
    error: str = None
    traceback: str = None

@router.post("/hackrx/run", response_model=HackrxRunResponse)
async def hackrx_run(request: HackrxRunRequest):
    url = request.documents
    # questions = request.questions  # Not used yet, but accepted
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download PDF: {response.status_code}")
            content_type = response.headers.get("content-type", "")
            if "application/pdf" not in content_type:
                raise HTTPException(status_code=400, detail="URL does not point to a PDF file (Content-Type is not application/pdf)")
            contents = response.content
        extracted_text, chunks, all_vectors = process_pdf(contents, url.split("/")[-1])
        if all_vectors is None:
            return JSONResponse(content={"status": "error", "detail": "PDF processing failed"}, status_code=500)
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e), "traceback": traceback.format_exc()}, status_code=500)
