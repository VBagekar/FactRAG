import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from usearch.index import Index
from factrag.corpus_builder import Passage, load_corpus
from factrag.bm25_retriever import RetrievedPassage


MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384  # dimension output of MiniLM


class DenseRetriever:
    def __init__(self):
        self.passages = []
        self.index = None
        self.model = None

    def build(self, corpus_path: str = "data/corpus.json",
              index_path: str = "data/dense_index.pkl",
              batch_size: int = 64):

        print(f"Loading model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.passages = load_corpus(corpus_path)

        print(f"Encoding {len(self.passages)} passages (this takes 2-3 mins)...")
        texts = [p.text for p in self.passages]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Normalize so dot product = cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-10, None)

        # usearch index — cosine metric, float32 vectors
        self.index = Index(ndim=VECTOR_DIM, metric="cos")
        keys = np.arange(len(embeddings), dtype=np.int64)
        self.index.add(keys, embeddings)

        Path(index_path).parent.mkdir(exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump({
                "passages": self.passages,
                "embeddings": embeddings,
            }, f)

        print(f"✅ Dense index built: {len(self.passages)} vectors")
        print(f"   Saved to {index_path}")

    def load(self, index_path: str = "data/dense_index.pkl"):
        print("Loading dense index...")
        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.passages = data["passages"]
        embeddings = data["embeddings"].astype(np.float32)
        self.model = SentenceTransformer(MODEL_NAME)

        self.index = Index(ndim=VECTOR_DIM, metric="cos")
        keys = np.arange(len(embeddings), dtype=np.int64)
        self.index.add(keys, embeddings)
        print(f"✅ Loaded {len(self.passages)} vectors")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(query_vec)
        query_vec = query_vec / np.clip(norm, 1e-10, None)

        matches = self.index.search(query_vec, top_k)

        # usearch returns a Matches object — flatten keys and distances to 1D arrays
        keys = np.array(matches.keys).flatten()
        distances = np.array(matches.distances).flatten()

        return [
            RetrievedPassage(
                passage=self.passages[int(keys[rank])],
                score=float(1 - distances[rank]),
                rank=rank + 1,
            )
            for rank in range(len(keys))
        ]