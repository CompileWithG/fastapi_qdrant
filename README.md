# FastAPI Qdrant PDF Embedding Service

## Overview
This service provides an API endpoint to upload PDF files, extract and chunk their text, and generate ONNX-accelerated embeddings for each chunk. It is built with FastAPI and leverages PyMuPDF, LangChain, and SentenceTransformers.

---

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd fastapi_qdrant
```

### 2. Create and Activate a Python Environment
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [venv]:
```sh
conda create -n fastapi_qdrant python=3.10 -y
conda activate fastapi_qdrant
# OR
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```sh
cd backend
pip install -r requirements.txt
```

> **Note:** Ensure only `PyMuPDF` (shows as `pymupdf` in `pip freeze`) is installed for PDF processing. Do not install the unrelated `fitz` package.

---

## Running the API Server

From the `backend` directory, start the FastAPI server:
```sh
uvicorn app.main:app --reload
```
- The API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Testing the PDF Upload Endpoint

### Option 1: Swagger UI (Recommended)
1. Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Find the `/api/upload-pdf` endpoint.
3. Click **"Try it out"**.
4. Upload a PDF file.
5. Click **"Execute"**.
6. View the JSON response with chunk/embedding stats and timings.

### Option 2: cURL
```sh
curl -X POST "http://127.0.0.1:8000/api/upload-pdf" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@sample.pdf"
```

### Option 3: Python Script
```python
import requests
url = "http://127.0.0.1:8000/api/upload-pdf"
with open("sample.pdf", "rb") as f:
    files = {"file": ("sample.pdf", f, "application/pdf")}
    response = requests.post(url, files=files)
print(response.json())
```

---

## Troubleshooting
- **ModuleNotFoundError: No module named 'fitz'**
  - Ensure `PyMuPDF` is installed (`pip install PyMuPDF`).
  - Do **not** install the unrelated `fitz` package from PyPI.
- **500 Internal Server Error**
  - Check the server logs for details. Common causes:
    - Invalid or corrupted PDF file.
    - Missing ONNX model file (`onnx/model_qint8_avx512.onnx`).
    - Dependency issues (see `requirements.txt`).
- **ONNX Model Not Found**
  - Ensure the required ONNX model file is present in the correct path, or update the path in the code.

---
