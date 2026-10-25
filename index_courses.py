import os
import re
import json
import logging
import warnings
from pypdf import PdfReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Silence warnings
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Lightweight, fast, high-quality embedding model (runs locally on your CPU/Metal)
MODEL_NAME = "all-MiniLM-L6-v2"

def extract_chunks_from_pdf(pdf_path, chunk_size=350, overlap=35):
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            full_text += f"\n[Page {page_num + 1}] " + text

        words = full_text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks
    except Exception:
        return []

def main():
    print(f"📦 Loading local embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    pdf_files = sorted([f for f in os.listdir(".") if f.endswith(".pdf")])
    knowledge_base = []

    print(f"📄 Found {len(pdf_files)} PDFs. Extracting text chunks...")
    for filename in tqdm(pdf_files):
        match = re.search(r"course(\d+)_(\d+)_(.+)\.pdf", filename, re.IGNORECASE)
        course = match.group(1) if match else "General"
        title = match.group(3).replace("_", " ") if match else filename

        chunks = extract_chunks_from_pdf(filename)
        for idx, chunk in enumerate(chunks):
            knowledge_base.append({
                "id": f"{filename}_{idx}",
                "file": filename,
                "title": title,
                "course": f"Course {course}",
                "text": chunk
            })

    print(f"⚡ Embedding {len(knowledge_base)} chunks locally on your Mac...")
    texts = [item["text"] for item in knowledge_base]
    
    # Generate embeddings locally (normalized for fast dot-product similarity)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    for item, emb in zip(knowledge_base, embeddings):
        item["embedding"] = emb.tolist()

    output_file = "knowledge_base.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False)

    print(f"\n🎉 Done! Created '{output_file}' locally without any 500 errors.")

if __name__ == "__main__":
    main()