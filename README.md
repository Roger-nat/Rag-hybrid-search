# Rag-hybrid-search
Hybrid-search RAG app: upload PDF/TXT/DOCX, ask questions, get cited answers. Combines ChromaDB vector search with BM25, fuses via RRF, reranks with a cross-encoder, then an LLM answers strictly from retrieved context or says "not found." Built with FastAPI, Streamlit, Docker.
