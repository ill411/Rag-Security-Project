# RAG Security Research Project

**Title:** How Secure Are Enterprise RAG Deployments?

**Author:** Ilhaan Ahmed Mohammed | Student No: A00047081

**Module:** MACS H6021 – Research Skills and Ethics | TU Dublin

## Project Overview
An empirical study measuring document extraction attack success rates 
against RAG-based LLM deployments, and evaluating a novel layered 
defence architecture (LEDA) against those attacks.

## Repository Structure
- corpus/ — 60 synthetic enterprise documents (4 types)
- prompts/ — adversarial and benign prompt sets
- results/ — raw experimental results in CSV format
- defences/ — LEDA defence implementation
- generate_corpus.py — corpus generation script
- rag_pipeline.py — core RAG pipeline

## Setup
1. Create virtual environment: python3 -m venv venv
2. Activate: source venv/bin/activate
3. Install packages: pip install -r requirements.txt
4. Add OpenAI API key to .env file
5. Run: python3 rag_pipeline.py

## Models Used
- Llama 3.1 8B (local via Ollama)
- GPT-4o mini (OpenAI API)

## Note
All documents in corpus/ are entirely synthetic and fictional.
No real personal or business data is used at any stage.