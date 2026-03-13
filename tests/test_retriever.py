import sys
sys.path.insert(0, '.')

from factrag.bm25_retriever import BM25Retriever

retriever = BM25Retriever()
retriever.build()

queries = [
    "What is India's population?",
    "How tall is Mount Everest?",
    "How many neurons does the human brain have?",
    "What is the speed of light?",
    "India GDP trillion economy",
]

for query in queries:
    print(f"\nQUERY: {query}")
    results = retriever.retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r.rank}] (score={r.score:.2f}) [{r.passage.title}]")
        print(f"       {r.passage.text[:120]}...")