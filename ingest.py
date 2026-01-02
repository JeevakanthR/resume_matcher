import os
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from simple_embeddings import SimpleHFEmbeddings


RESUME_FOLDER = "resumes"
DB_FOLDER = "vector_db"

# CPU‑safe embedding model
embeddings = SimpleHFEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def read_pdf(path: str) -> str:
    """Read a PDF file and return extracted text."""
    reader = PdfReader(path)
    full_text: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            full_text.append(text)
    return "\n".join(full_text)


def main():
    if not os.path.exists(RESUME_FOLDER):
        os.makedirs(RESUME_FOLDER, exist_ok=True)

    texts = []
    metadatas = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    for file in os.listdir(RESUME_FOLDER):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(RESUME_FOLDER, file)
        content = read_pdf(path)
        if not content.strip():
            continue

        chunks = splitter.split_text(content)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": file})

    if not texts:
        print("⚠️ No PDF text found in the resumes folder. Please add valid PDF resumes and try again.")
        return

    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_FOLDER,
    )

    print("✅ All resumes indexed successfully!")


if __name__ == "__main__":
    main()
