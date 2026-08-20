# Multimodal RAG System

A multimodal Retrieval-Augmented Generation (RAG) system for querying PDF documents using text and image information.

## Features

- PDF document processing
- Text extraction and chunking
- Image extraction
- CLIP-based embeddings
- FAISS vector similarity search
- Gemini-powered answer generation
- Streamlit interface
- Automated evaluation pipeline

## Architecture

PDF
↓
PyMuPDF
↓
Text + Image Extraction
↓
Text Chunking
↓
CLIP Embeddings
↓
FAISS
↓
Top-K Retrieval
↓
Gemini
↓
Answer

## Tech Stack

- Python
- LangChain
- PyMuPDF
- Hugging Face
- CLIP
- FAISS
- Google Gemini
- Streamlit

## Current Version

V1 — Baseline Multimodal RAG

### V1 Baseline

- Top-K: 5
- Top-5 Retrieval Accuracy: 80%
- Answer Accuracy: 65.5%
- Image Retrieval Accuracy: 0%
- Average Retrieval Time: 39.5 ms
- Average Generation Time: 1.21 s
- Average End-to-End Latency: 1.25 s

## Future Improvements

### V2
- Improved multimodal retrieval
- Hybrid text + image retrieval
- Candidate expansion
- Reranking

### V3
- Cloud vector database
- Scalable document processing
- LangSmith evaluation and tracing
- Production deployment
