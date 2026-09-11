# DocVision AI — Multimodal RAG

[🚀 Live Demo](https://docvision-ai-v2.streamlit.app/) | [💻 GitHub](https://github.com/samarsingh04999-bit/DocVision-AI)

A multimodal Retrieval-Augmented Generation (RAG) system for querying PDF documents using both text and image information.

## Overview

DocVision AI processes PDF documents, extracts text and images, generates multimodal embeddings using CLIP, retrieves relevant content using FAISS, and uses Google Gemini to generate grounded answers.

The project was developed iteratively from a V1 baseline to V2 with improved multimodal retrieval.

## Features

- PDF text and image extraction using PyMuPDF
- Text chunking with LangChain
- CLIP-based text and image embeddings
- FAISS vector similarity search
- Separate text and image retrieval
- Multimodal context construction
- Gemini-powered document Q&A
- Streamlit interface
- Automated 30-question evaluation pipeline

## Architecture

```text
                    PDF
                     ↓
                  PyMuPDF
                     ↓
             Text + Images
                     ↓
               CLIP Embeddings
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
     Text FAISS            Image FAISS
          ↓                     ↓
       Top-5                  Top-5
          └──────────┬──────────┘
                     ↓
             Multimodal Context
                     ↓
               Google Gemini
                     ↓
                  Answer
V1 — Baseline

V1 used a unified FAISS index for text and image embeddings.

Metric	V1
Top-K	5
Retrieval Accuracy	80%
Answer Accuracy	65.5%
Image Retrieval Accuracy	0%
Retrieval Latency	39.5 ms
End-to-End Latency	1.25 s
V1 Limitation

The main limitation was poor image retrieval. Relevant images were not being retrieved reliably for image-based queries.

V2 — Improved Multimodal Retrieval

V2 introduced separate FAISS indexes for text and images.

Text Top-K: 5
Image Top-K: 5
Improved multimodal retrieval coverage
Dedicated evaluation pipeline
Retrieval and generation latency measurement
V2 Results
Metric	V1	V2
Answer Accuracy	65.5%	~73.3%
Image Retrieval	0%	100%*
Retrieval Latency	39.5 ms	49.7 ms

*V2 image retrieval measures whether an image was retrieved for image/multimodal questions, not whether every retrieved image was the correct relevant image.

Tech Stack

Python · LangChain · PyMuPDF · Hugging Face CLIP · FAISS · Google Gemini · Streamlit

Project Structure
DocVision-AI/
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── evaluation/
    ├── evaluate.py
    └── questions.json
V3 — Next Steps

The next version will focus on improving retrieval quality and moving toward a production-ready architecture.

Reranking retrieved candidates
Recall@K / Precision@K evaluation
Cloud vector database
FastAPI backend
LangSmith tracing and evaluation
Scalable document processing
Production deployment

