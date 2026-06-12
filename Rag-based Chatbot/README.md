# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about YouTube videos using their transcripts. The project extracts video transcripts, splits them into chunks, generates embeddings using open-source models, stores them in FAISS, and retrieves relevant context to answer user queries with a local LLM.

## Features

* Extract YouTube video transcripts
* Chunk and process transcript text
* Generate embeddings with open-source models
* Store and retrieve documents using FAISS
* Answer questions using RAG
* Fully local and free 

## Tech Stack

* LangChain
* FAISS
* Hugging Face Embeddings ( MiniLM)
* Ollama (Phi-3)
* YouTube Transcript API

## Workflow

YouTube Video → Transcript → Text Splitting → Embeddings → FAISS → Retriever → Phi-3 → Answer

## Installation

```bash
pip install langchain langchain-community langchain-huggingface
pip install langchain-ollama faiss-cpu
pip install sentence-transformers youtube-transcript-api
```

## Usage

1. Provide a YouTube video ID.
2. Generate the vector database from the transcript.
3. Ask questions about the video content.
4. Receive context-aware answers generated through RAG.

## License

This project is intended for educational and research purposes.
