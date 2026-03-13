import sys
sys.path.insert(0, '.')

from factrag.hybrid_retriever import HybridRetriever
from factrag.reader import Reader

queries = [
    "What is India's population?",
    "How tall is Mount Everest?",
    "How many neurons does the human brain have?",
    "What is the speed of light?",
    "What is India's GDP?",
]

hybrid = HybridRetriever()
hybrid.load()

reader = Reader()

print("\n" + "=" * 60)
print("FactRAG — Reader Output")
print("=" * 60)

for query in queries:
    print(f"\nQ: {query}")
    passages = hybrid.retrieve(query, top_k=5)
    answer = reader.extract_answer(query, passages, top_k=3)

    if answer:
        print(f"  A: {answer.text}")
        print(f"     confidence={answer.score} | source: [{answer.source_title}]")
        print(f"     passage: {answer.source_passage[:100]}...")
    else:
        print("  A: No answer found")