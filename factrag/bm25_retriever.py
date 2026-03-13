import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from factrag.corpus_builder import Passage, load_corpus


@dataclass
class RetrievedPassage:
    passage: Passage
    score: float
    rank: int


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self):
        self.passages = []
        self.bm25 = None

    def build(self, corpus_path: str = "data/corpus.json",
              index_path: str = "data/bm25_index.pkl"):

        self.passages = load_corpus(corpus_path)
        print(f"Building BM25 index over {len(self.passages)} passages...")

        tokenized = [_tokenize(p.text) for p in self.passages]
        self.bm25 = BM25Okapi(tokenized)

        # Save index so we don't rebuild every run
        Path(index_path).parent.mkdir(exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "passages": self.passages}, f)

        print(f"✅ BM25 index saved to {index_path}")

    def load(self, index_path: str = "data/bm25_index.pkl"):
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.passages = data["passages"]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)),
                             key=lambda i: scores[i],
                             reverse=True)[:top_k]

        return [
            RetrievedPassage(
                passage=self.passages[i],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_indices)
        ]