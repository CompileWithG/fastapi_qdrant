# app/api/endpoints/upload.py
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.services.pdf_processor import PDFProcessor
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

router = APIRouter()

# Initialize processor
processor = PDFProcessor()

# --- Security Setup ---
security = HTTPBearer()

# Get token from .env
API_TOKEN = os.getenv("API_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Validate the Authorization Bearer Token against the API_TOKEN from .env
    - Checks if API_TOKEN is configured
    - Validates the incoming token matches exactly
    - Returns HTTP 401 for invalid tokens
    - Returns HTTP 500 if server is misconfigured
    """
    if not API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API_TOKEN not set in environment",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    incoming_token = credentials.credentials
    
    if not incoming_token or incoming_token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True

# --- Request Model ---
class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Secure Endpoint ---
@router.post("/hackrx/run")
async def process_document(
    request: DocumentRequest,
    token_verified: bool = Depends(verify_token)  # Enforces authentication
):
    """
    Secured endpoint to process document and questions
    - Requires valid Bearer token in Authorization header
    - Processes the document and questions if authorized
    """
    try:
        return processor.process(request.documents, request.questions)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )