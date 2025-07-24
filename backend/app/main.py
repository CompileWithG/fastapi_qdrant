from fastapi import FastAPI
from app.api.endpoints import upload

app = FastAPI(title="PDF API")

app.include_router(upload.router, prefix="/api")
