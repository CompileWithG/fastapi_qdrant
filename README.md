refactor(pdf_processor): added endpoint-based mission brief handling with caching fallback

- Implemented detection of mission briefs containing GET/POST endpoints
- Added logic to route such documents through run_doc_with_endpoints for GPT-guided execution
- Preserved normal vector-search flow for non-endpoint documents
