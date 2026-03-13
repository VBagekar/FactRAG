from dataclasses import dataclass
from factrag.bm25_retriever import BM25Retriever, RetrievedPassage
from factrag.dense_retriever import DenseRetriever
from factrag.corpus_builder import Passage


RRF_K = 60


class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

    def build(self, corpus_path: str = "data/corpus.json"):
        self.bm25.build(corpus_path)
        self.dense.build(corpus_path)

    def load(self,
             bm25_path: str = "data/bm25_index.pkl",
             dense_path: str = "data/dense_index.pkl"):
        self.bm25.load(bm25_path)
        self.dense.load(dense_path)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        # Get top-k*2 from each retriever so fusion has enough candidates
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)

        # Build passage_id → RRF score map
        rrf_scores: dict[str, float] = {}

        for result in bm25_results:
            pid = result.passage.id
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (RRF_K + result.rank)

        for result in dense_results:
            pid = result.passage.id
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (RRF_K + result.rank)

        # Build a lookup so we can retrieve Passage objects by id
        passage_lookup: dict[str, Passage] = {}
        for result in bm25_results + dense_results:
            passage_lookup[result.passage.id] = result.passage

        # Sort by RRF score descending and return top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievedPassage(
                passage=passage_lookup[pid],
                score=round(score, 6),
                rank=rank + 1,
            )
            for rank, (pid, score) in enumerate(ranked[:top_k])
        ]