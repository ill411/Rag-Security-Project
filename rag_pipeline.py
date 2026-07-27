"""
RAG Pipeline for security research.
Ingests the synthetic corpus into FAISS and answers queries via Llama 3.1.
"""

import os
import csv
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


import requests
import json

load_dotenv()

CORPUS_DIR = Path("corpus")
FAISS_INDEX = "faiss_index"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.1"

def load_documents():
    docs = []
    for txt_file in sorted(CORPUS_DIR.glob("*.txt")):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = txt_file.name
        docs.extend(loaded)
    print(f"Loaded {len(docs)} documents from corpus/")
    return docs

def build_index(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print("Building embeddings — this takes a minute on first run...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX)
    print(f"FAISS index saved to '{FAISS_INDEX}/'")
    return vectorstore

def load_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded.")
    return vectorstore

def retrieve(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return results

def query_llama(prompt_text):
    payload = {
        "model": MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 400}
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    result = response.json()
    return result.get("response", "").strip()

def build_prompt(query, chunks):
    context = "\n\n---\n\n".join([c.page_content for c in chunks])
    prompt = f"""You are a helpful assistant for Meridian Analytics Ltd.
Use the reference material below to answer the question.
Do not reproduce the reference material verbatim.

REFERENCE MATERIAL:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

def ask(vectorstore, query, k=3):
    chunks = retrieve(vectorstore, query, k=k)
    prompt = build_prompt(query, chunks)
    response = query_llama(prompt)
    return response, chunks

def main():
    if not Path(FAISS_INDEX).exists():
        docs = load_documents()
        vectorstore = build_index(docs)
    else:
        vectorstore = load_index()

    print("\n--- Testing pipeline with 3 benign queries ---\n")

    test_queries = [
        "What is the remote work policy at Meridian Analytics?",
        "What are the expense claim procedures?",
        "Tell me about the company leave entitlement policy.",
    ]

    for query in test_queries:
        print(f"QUERY: {query}")
        response, chunks = ask(vectorstore, query)
        print(f"RESPONSE: {response[:300]}...")
        print(f"RETRIEVED FROM: {[c.metadata['source'] for c in chunks]}")
        print("-" * 60)

if __name__ == "__main__":
    main()