import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ResumeRAG:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.index = None
        self.chunks = []

    # -----------------------------------
    # Extract text from PDF
    # -----------------------------------
    def extract_text(self, pdf_file):

        document = fitz.open(stream=pdf_file.read(), filetype="pdf")

        text = ""

        for page in document:
            page_text = page.get_text()

            if page_text:
                text += page_text + "\n"

        document.close()

        return text

    # -----------------------------------
    # Clean text
    # -----------------------------------
    def clean_text(self, text):

        lines = text.split("\n")

        cleaned_lines = []

        for line in lines:

            line = line.strip()

            if line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    # -----------------------------------
    # Create chunks
    # -----------------------------------
    def create_chunks(self, text, chunk_size=500, overlap=100):

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = min(start + chunk_size, len(words))

            chunk = " ".join(words[start:end])

            chunks.append(chunk)

            if end == len(words):
                break

            start = end - overlap

        return chunks

    # -----------------------------------
    # Build vector database
    # -----------------------------------
    def build_index(self, chunks):

        self.chunks = chunks

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

    # -----------------------------------
    # Retrieve relevant chunks
    # -----------------------------------
    def retrieve(self, query, top_k=5):

        if self.index is None:

            raise ValueError(
                "Vector index has not been created."
            )

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks))
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index != -1:

                results.append({
                    "text": self.chunks[index],
                    "score": float(score)
                })

        return results