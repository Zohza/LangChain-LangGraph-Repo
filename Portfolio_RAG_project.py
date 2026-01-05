# rag_cv.py

import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import os

# -----------------------------
# 1️⃣ Read PDF function
# -----------------------------
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# -----------------------------
# 2️⃣ Split text into chunks
# -----------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# 3️⃣ Setup Chroma client & collection
# -----------------------------
client = chromadb.PersistentClient(path="./cv_rag_db")

# Sentence-Transformers embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Use a fresh collection (change name if rerunning to avoid old metadata issues)
collection_cv = client.get_or_create_collection(
    name="cv_data_v2",
    embedding_function=sentence_transformer_ef
)

# -----------------------------
# 4️⃣ Load CV and add to collection
# -----------------------------
cv_text = read_pdf("cv.pdf")
cv_chunks = chunk_text(cv_text)

# IDs and metadata
ids = [f"cv_chunk_{i}" for i in range(len(cv_chunks))]
metadatas = [{"chunk_id": i} for i in range(len(cv_chunks))]

# Add to collection
collection_cv.add(
    documents=cv_chunks,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Collection created with {collection_cv.count()} chunks")

# -----------------------------
# 5️⃣ Query function (RAG)
# -----------------------------
def search_cv(query, n_results=3):
    results = collection_cv.query(
        query_texts=[query],
        n_results=n_results
    )

    print(f"🔎 Query: {query}\n")
    for i, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        print(f"ID: {i}")
        print(f"Distance: {dist:.2f}")
        print(f"Document chunk:\n{doc}")
        print(f"Metadata: {meta}")
        print("----")

# -----------------------------
# 6️⃣ Test queries
# -----------------------------
if __name__ == "__main__":
    search_cv("work experience")
    search_cv("programming skills")
    search_cv("education history")
