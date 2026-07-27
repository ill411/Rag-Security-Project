"""
Rebuilds FAISS index with metadata-aware chunking
so retrieval works correctly across all document types.
"""
import time
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CORPUS_DIR = Path("corpus")
FAISS_INDEX = "faiss_index"

def main():
    docs = []
    for txt_file in sorted(CORPUS_DIR.glob("*.txt")):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = txt_file.name
            fname = txt_file.stem
            if fname.startswith("POLICY"):
                doc.metadata["doc_type"] = "policy"
            elif fname.startswith("FINANCIAL"):
                doc.metadata["doc_type"] = "financial"
            elif fname.startswith("PROJECT"):
                doc.metadata["doc_type"] = "project_spec"
            elif fname.startswith("PERSONAL"):
                doc.metadata["doc_type"] = "personal"
        docs.extend(loaded)
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print("Building embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX)
    print(f"Index saved to {FAISS_INDEX}/")

    print("\nTesting retrieval across all types...")
    tests = [
        ("POLICY-011",      "remote work eligibility policy POL-4770"),
        ("FINANCIAL-002",   "quarterly revenue financial summary FIN"),
        ("PROJECT_SPEC-005","project Atlas PROJ objectives timeline"),
        ("PERSONAL-001",    "employee HR record EMP salary band"),
    ]
    for expected, query in tests:
        results = vectorstore.similarity_search(query, k=3)
        sources = [r.metadata["source"] for r in results]
        hit = any(expected.split("-")[0] in s for s in sources)
        print(f"Query: {query[:40]}")
        print(f"Retrieved: {sources}")
        print(f"Correct type retrieved: {'YES' if hit else 'NO'}")
        print()

if __name__ == "__main__":
    main()