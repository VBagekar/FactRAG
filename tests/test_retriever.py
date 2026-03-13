import sys
sys.path.insert(0, '.')

from factrag.bm25_retriever import BM25Retriever
from factrag.dense_retriever import DenseRetriever

queries = [
    "What is India's population?",
    "How tall is Mount Everest?",
    "How many neurons does the human brain have?",
    "What is the speed of light?",
    "India GDP trillion economy",
]

print("=" * 60)
print("BM25 RETRIEVER")
print("=" * 60)

bm25 = BM25Retriever()
bm25.build()

for query in queries:
    print(f"\nQUERY: {query}")
    results = bm25.retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r.rank}] (score={r.score:.2f}) [{r.passage.title}]")
        print(f"       {r.passage.text[:120]}...")

print("\n" + "=" * 60)
print("DENSE RETRIEVER")
print("=" * 60)

dense = DenseRetriever()
dense.load()   # already built, just load

for query in queries:
    print(f"\nQUERY: {query}")
    results = dense.retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r.rank}] (score={r.score:.3f}) [{r.passage.title}]")
        print(f"       {r.passage.text[:120]}...")