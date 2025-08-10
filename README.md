# Intelligent Document Processing System

## Overview

This system is an advanced document processing pipeline that intelligently handles various document types (PDFs, DOCX, PowerPoint, emails, images, webpages) and answers questions about their content. The system uses multiple AI models, vector databases, and intelligent routing to provide accurate answers while optimizing performance through sophisticated caching mechanisms.

## 🏗️ Architecture Overview

The system consists of several key components:

1. **Document Processing Pipeline** - Extracts and processes content from various file types
2. **Intelligent Routing System** - Determines the best processing approach based on document type and content
3. **Vector Database Integration** - Uses Qdrant for semantic search and retrieval
4. **Multi-layered Caching System** - Optimizes performance and reduces redundant processing
5. **LLM Integration** - Uses GPT models for analysis, routing decisions, and answer generation
6. **Agent-based Processing** - Handles dynamic content requiring API calls and multi-step operations

## 🔄 Processing Workflows

### 1. Static Document Workflow (Traditional RAG)

**Used for:** Regular PDFs, Word documents, PowerPoint files, images, emails - documents with static content that can be answered from the content alone.

**Process Flow:**
```
Document URL → File Type Detection → Download & Extract → Text Chunking → 
Embeddings Generation → Vector Storage → Question Processing → 
Semantic Search → Answer Generation → Cache Results
```

**Detailed Steps:**
1. **File Type Detection**: Uses GPT-3.5-turbo to intelligently determine file type from URL
2. **Content Extraction**: Specialized extractors for each file type:
   - PDFs: PyMuPDF (fitz) for text extraction
   - Word docs: python-docx for structured text
   - PowerPoint: python-pptx + OCR for images within slides
   - Images: Tesseract OCR for text recognition
   - Emails: Email parser for structured email content
3. **Text Chunking**: RecursiveCharacterTextSplitter (768 chars, no overlap)
4. **Embedding Generation**: SentenceTransformer (all-MiniLM-L6-v2) with ONNX optimization
5. **Vector Storage**: Qdrant with HNSW indexing and scalar quantization
6. **Question Processing**: Convert questions to embeddings, perform semantic search
7. **Answer Generation**: GPT-4 processes retrieved context to generate answers

### 2. Dynamic Document Workflow (Agent-based)

**Used for:** Documents containing instructions for API calls, multi-step processes, or webpages requiring dynamic interaction.

**Process Flow:**
```
Document URL → Content Analysis → Dynamic Pattern Detection → 
Agent Initialization → Tool Setup → Multi-step Execution → 
API Calls → Response Processing → Final Answer Generation
```

**Detailed Steps:**
1. **Dynamic Content Detection**: Pattern matching and LLM analysis to identify:
   - API call instructions (`call GET https://...`, `make a request to...`)
   - Multi-step conditional processes
   - External service integrations
   - Dynamic data dependencies

2. **Agent Initialization**: Sets up LangChain ZeroShotReactDescription agent with:
   - GPT-4 as the reasoning engine
   - Custom tool suite for various operations
   - Error handling and retry mechanisms

3. **Tool Execution**: Agent uses available tools to:
   - Make HTTP GET/POST requests
   - Parse HTML content
   - Search document content
   - Map data between different sources

4. **Multi-step Processing**: Agent follows document instructions to:
   - Execute API calls in correct sequence
   - Process responses and extract needed information
   - Apply conditional logic based on results

### 3. Webpage Processing Workflow

**Used for:** Live web content that needs to be accessed and processed in real-time.

**Process Flow:**
```
URL → Agent Activation → HTTP Request → HTML Parsing → 
Content Extraction → Question-specific Processing → Answer Generation
```

## 🧠 Intelligent Routing System

The system automatically determines which workflow to use based on:

### File Type Analysis
```python
# Uses GPT-3.5-turbo for intelligent file type detection
prompt = "Given this URL, determine what type of file it points to..."
```

### Dynamic Content Detection
The system scans for patterns indicating dynamic content:
- Regex patterns for API calls and HTTP requests
- Step-by-step instruction detection  
- Conditional logic patterns
- External service references

### LLM-based Analysis
For ambiguous cases, GPT-3.5-turbo analyzes document content:
```python
prompt = """
Analyze this document content and questions to determine if they require 
DYNAMIC ACTIONS like making HTTP requests, calling APIs, or executing 
multi-step processes.
"""
```

## 🗄️ Multi-layered Caching System

### 1. Document Cache (`document_cache.json`)
- **Purpose**: Maps document URLs to Qdrant collection IDs
- **Benefits**: Avoids reprocessing same documents
- **Structure**: `{url: collection_id}`

### 2. Q&A Cache (`qa_cache.json`)  
- **Purpose**: Stores question-answer pairs for processed documents
- **Benefits**: Instant responses for repeated questions
- **Structure**: `{collection_id: {question: answer}}`

### 3. Text Cache (`text_cache.json`)
- **Purpose**: Stores extracted text content from documents
- **Benefits**: Avoids re-downloading and re-extracting content
- **Structure**: `{url: extracted_text}`

### 4. Dynamic Documents Cache (`dynamic_documents.json`)
- **Purpose**: Tracks which documents require agent-based processing
- **Benefits**: Immediate routing decisions for known dynamic content
- **Structure**: `{url: is_dynamic_boolean}`

### Cache Intelligence
- **Static Documents**: Full caching (document + Q&A + text)
- **Dynamic Documents**: Limited caching (text only, no Q&A caching)
- **Webpages**: No caching (always fetch fresh content)

## 🤖 LLM Integration & Function Calls

### 1. File Type Detection (GPT-3.5-turbo)
```python
# Lightweight model for quick file type determination
model="gpt-3.5-turbo", max_tokens=10, temperature=0
```

### 2. Dynamic Content Analysis (GPT-3.5-turbo)  
```python
# Analyzes if document requires agent-based processing
model="gpt-3.5-turbo", max_tokens=10, temperature=0
```

### 3. Agent Reasoning (GPT-4)
```python
# Powers the LangChain agent for complex multi-step operations
ChatOpenAI(model="gpt-4", temperature=0)
```

### 4. Answer Generation (GPT-4)
```python
# Final answer synthesis from retrieved context
model="gpt-4.1", max_tokens=4000, temperature=0
```

### Function Calling Tools

The agent system includes 6 specialized tools:

#### 1. HTTP GET Tool
```python
def http_get_func(url: str) -> str:
    """Make HTTP GET request and return response text"""
```

#### 2. HTTP POST Tool  
```python
def http_post_func(url_and_data: str) -> str:
    """Make HTTP POST request with data and headers"""
    # Format: 'URL|||DATA|||HEADERS_JSON'
```

#### 3. HTML Extraction Tool
```python
def extract_from_html_func(html_and_params: str) -> str:
    """Extract content from HTML using BeautifulSoup"""
    # Format: 'HTML|||ELEMENT_TYPE|||ELEMENT_ID|||ELEMENT_CLASS'
```

#### 4. Document Content Tool
```python
def get_document_content_func(dummy_input: str = "") -> str:
    """Get full content of currently processed document"""
```

#### 5. Document Search Tool
```python
def search_document_content_func(search_term: str) -> str:
    """Search for specific content within document"""
```

#### 6. City-Landmark Mapping Tool
```python
def find_city_landmark_func(city_name: str) -> str:
    """Find landmark associated with specific city from document tables"""
```

## 📊 Vector Database Configuration (Qdrant)

### Collection Setup
```python
vectors_config=VectorParams(
    size=384,  # SentenceTransformer embedding dimension
    distance=Distance.COSINE,
    on_disk=False
)
```

### Performance Optimizations
- **HNSW Indexing**: Fast approximate nearest neighbor search
- **Scalar Quantization**: INT8 quantization for memory efficiency  
- **Segmentation**: 8 default segments for parallel processing
- **WAL Configuration**: 1024MB capacity with 2 segments ahead

### Search Configuration
```python
search_params=models.SearchParams(
    hnsw_ef=32,  # Search accuracy parameter
    exact=False  # Use approximate search for speed
)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Qdrant vector database (local or cloud)
- Tesseract OCR (for image processing)

### Installation Steps

1. **Clone the Repository**
```bash
git clone <repository-url>
cd document-processing-system
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install System Dependencies**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
```

5. **Setup Qdrant Database**

**Option A: Docker (Recommended)**
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Installation**
```bash
# Follow instructions at: https://qdrant.tech/documentation/install/
```

6. **Environment Configuration**
Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://localhost:6333
```

7. **Verify Installation**
```python
from app.services.pdf_processor import PDFProcessor
processor = PDFProcessor()
print("System initialized successfully!")
```

### Basic Usage

```python
import asyncio
from app.services.pdf_processor import PDFProcessor

async def main():
    processor = PDFProcessor()
    
    # Process a document with questions
    result = await processor.process(
        document_url="https://example.com/document.pdf",
        questions=[
            "What is the main topic of this document?",
            "What are the key findings mentioned?"
        ]
    )
    
    print("Answers:", result["answers"])

# Run the example
asyncio.run(main())
```

## 🔍 Performance Characteristics

### Static Documents
- **First Processing**: 5-15 seconds (depending on document size)
- **Cached Questions**: <1 second (instant retrieval)
- **New Questions on Cached Document**: 2-5 seconds

### Dynamic Documents  
- **Processing Time**: 10-30 seconds (depends on API calls and complexity)
- **No Q&A Caching**: Always processes fresh (ensures dynamic accuracy)

### Memory Usage
- **Embedding Model**: ~50MB (ONNX optimized)
- **Vector Storage**: Varies by document size
- **Cache Files**: Minimal disk usage

## 🎯 Use Cases & Examples

### 1. Static PDF Analysis
```python
# Financial report analysis
result = await processor.process(
    "https://company.com/annual-report.pdf",
    ["What was the revenue growth?", "What are the main risks?"]
)
```

### 2. Dynamic API Integration
```python
# Document with API instructions
result = await processor.process(
    "https://api-docs.com/instructions.pdf",
    ["What is the current flight status for route ABC?"]
)
# System automatically detects API calls needed and executes them
```

### 3. Multi-format Processing
```python
# PowerPoint with embedded images
result = await processor.process(
    "https://company.com/presentation.pptx",
    ["What are the key metrics shown in the charts?"]
)
# Automatically uses OCR for image-based content
```

## 🔧 Configuration Options

### Chunk Size Adjustment
```python
# In chunk_text method
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=768,  # Adjust based on your needs
    chunk_overlap=0,
    separators=["\n\n", "\n", "."]
)
```

### Embedding Model Selection
```python
# In _initialize_embedder method
self.embedder = SentenceTransformer(
    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    backend="onnx",  # For speed optimization
)
```

### Search Parameters Tuning
```python
# In search_questions method
search_results = self.qdrant_client.search(
    limit=8,  # Number of relevant chunks to retrieve
    search_params=models.SearchParams(hnsw_ef=32)
)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Ensure Qdrant is running on port 6333
   - Check firewall settings
   - Verify Docker container status

2. **OpenAI API Errors**
   - Verify API key in .env file
   - Check API quota and billing
   - Ensure proper model access permissions

3. **Tesseract OCR Issues**
   - Install language packs: `sudo apt-get install tesseract-ocr-<lang>`
   - Verify installation: `tesseract --version`
   - Check image quality for OCR accuracy

4. **Memory Issues**
   - Reduce batch_size in embedding generation
   - Adjust chunk_size for large documents
   - Monitor Qdrant memory usage

### Performance Optimization

1. **For Large Documents**
   - Increase chunk_size to reduce number of vectors
   - Use on_disk=True for Qdrant storage
   - Implement document preprocessing

2. **For High Query Volume**
   - Enable Qdrant clustering
   - Implement connection pooling
   - Use async processing for concurrent requests

3. **For Low Memory Systems**
   - Use quantization for embeddings
   - Reduce vector dimensions
   - Implement batch processing

## 📈 Monitoring & Logging

The system includes built-in performance monitoring:

```python
def print_elapsed_time(self, message=""):
    elapsed = time.time() - getattr(self, "start_time", time.time())
    print(f"[+{elapsed:.2f}s] {message}")
```

Logs are written to `logs.txt` for answer tracking and debugging.

## 🔒 Security Considerations

1. **API Key Management**: Use environment variables, never commit keys
2. **Input Validation**: URLs and questions are validated before processing
3. **Rate Limiting**: Implement rate limiting for production use
4. **Sandboxing**: Agent tools are designed with safety constraints
5. **Data Privacy**: Cache files may contain sensitive information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include system information and error logs
