# LangChain Learning Journey

This repository contains my hands-on practice and projects while learning LangChain. The code follows along with tutorials, experiments, and mini-projects covering the core concepts required to build LLM-powered applications.

## Topics Covered

### Fundamentals
- Prompt Templates
- Chat Models
- Message Types
- Output Parsers
- Structured Outputs
- Pydantic Integration

### LCEL (LangChain Expression Language)
- RunnableSequence
- RunnableParallel
- RunnablePassthrough
- RunnableLambda
- RunnableBranch

### Chains
- Sequential Chains
- Parallel Chains
- Conditional Chains
- Prompt Chaining

### Working with Local LLMs
- Ollama Integration
- Phi-3
- Qwen Models
- Local Inference Workflows

### Document Processing
- TextLoader
- PyPDFLoader
- DirectoryLoader
- Document Splitting
- Chunking Strategies

### Embeddings & Vector Databases
- Hugging Face Embeddings
- FAISS
- ChromaDB
- Similarity Search
- Max Marginal Relevance (MMR)

### Retrieval-Augmented Generation (RAG)
- Retriever Creation
- Context Retrieval
- Question Answering over Documents
- YouTube Transcript RAG Chatbot

### Tool Calling & Agents
- Custom Tools
- Calculator Tool
- Student Assistant Agent
- Tool Binding with Ollama

## Technologies Used

- Python
- LangChain
- Ollama
- Phi-3
- Qwen
- Hugging Face Embeddings
- FAISS
- ChromaDB
- Pydantic

## Projects

### YouTube RAG Chatbot
A chatbot that answers questions about YouTube videos by:
1. Extracting transcripts
2. Creating embeddings
3. Storing vectors in FAISS
4. Retrieving relevant context
5. Generating answers using a local LLM

### Student Assistant Agent
A simple AI agent capable of:
- Listing students
- Retrieving student information
- Adding students
- Deleting students

### Calculator Tool Agent
A tool-calling application that performs mathematical operations through custom LangChain tools.

## Learning Outcome

Through this repository, I learned how to:

- Build LLM applications using LangChain
- Create reusable chains with LCEL
- Work with local LLMs using Ollama
- Build RAG systems from scratch
- Use vector databases and embeddings
- Create custom tools and agents
- Process and retrieve information from documents

## Repository Purpose

This repository serves as my personal LangChain learning notebook and implementation archive as I progress toward building more advanced AI, RAG, and Agentic applications.
