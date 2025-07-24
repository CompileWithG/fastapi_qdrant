# app/api/endpoints/upload.py
from fastapi import APIRouter, HTTPException
from app.services.pdf_processor import PDFProcessor
from pydantic import BaseModel
from typing import List

router = APIRouter()

# Initialize processor
processor = PDFProcessor()

class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

@router.post("/hackrx/run")
async def process_document(request: DocumentRequest):
    """Endpoint to process document and questions"""
    try:
        processor.process(request.documents, request.questions)
        return {"status": "processing completed"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
