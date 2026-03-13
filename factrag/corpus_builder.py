
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import wikipediaapi

@dataclass
class Passage:
    id: str           # unique ID e.g. "india_0", "india_1"
    title: str        # Wikipedia article title e.g. "India"
    text: str         # the actual chunk text
    word_count: int   # number of words in this chunk
    source_url: str   # Wikipedia URL for citation


TOPICS = {
    "geography": [
        "India", "China", "United States", "Amazon River",
        "Mount Everest", "Sahara Desert", "Pacific Ocean",
    ],
    "science": [
        "Human brain", "Solar System", "Speed of light",
        "DNA", "Black hole", "Periodic table",
    ],
    "economics": [
        "Gross domestic product", "World Bank",
        "International Monetary Fund", "Stock market",
        "Inflation", "Indian economy",
    ],
    "sports": [
        "Cricket", "Virat Kohli", "FIFA World Cup",
        "Olympic Games", "Indian Premier League",
    ],
    "history": [
        "World War II", "Indian independence movement",
        "Space Race", "Industrial Revolution",
    ],
}




def _chunk_text(text: str,
                chunk_size: int = 100,
                stride: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    
    Args:
        text:       raw article text
        chunk_size: words per chunk (100 ≈ 2-3 sentences)
        stride:     advance by this many words each step
                    (stride < chunk_size = overlap)
    
    Returns:
        List of text chunks
    """
    # Clean text — remove extra whitespace and Wikipedia markup artifacts
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'==+[^=]+=+', '', text)  # remove section headers
    
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 20:  # skip tiny chunks at the end
            chunks.append(chunk)
        i += stride

    return chunks



def _fetch_article(wiki: wikipediaapi.Wikipedia,
                   title: str) -> Optional[str]:
    """
    Fetch the full text of a Wikipedia article.
    Returns None if article not found.
    """
    page = wiki.page(title)
    if not page.exists():
        print(f"  ⚠️  Article not found: '{title}'")
        return None
    return page.text


def build_corpus(output_path: str = "data/corpus.json",
                 chunk_size: int = 100,
                 stride: int = 50) -> list[Passage]:
    """
    Fetch Wikipedia articles, chunk them, and save to JSON.

    Args:
        output_path: where to save the corpus JSON
        chunk_size:  words per chunk
        stride:      overlap stride (stride < chunk_size = overlap)

    Returns:
        List of Passage objects
    """
    
    wiki = wikipediaapi.Wikipedia(
        user_agent="FactRAG/1.0 (research project)",
        language="en",
    )

    all_passages = []
    total_articles = sum(len(v) for v in TOPICS.values())
    processed = 0

    print(f"\n📚 Building corpus from {total_articles} Wikipedia articles...\n")

    for domain, titles in TOPICS.items():
        print(f"  Domain: {domain.upper()}")

        for title in titles:
            print(f"    Fetching: '{title}'...", end=" ")
            text = _fetch_article(wiki, title)

            if text is None:
                continue

            # Chunk the article
            chunks = _chunk_text(text, chunk_size, stride)

            # Build Passage objects
            base_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            for i, chunk in enumerate(chunks):
                passage = Passage(
                    id=f"{title.lower().replace(' ', '_')}_{i}",
                    title=title,
                    text=chunk,
                    word_count=len(chunk.split()),
                    source_url=base_url,
                )
                all_passages.append(passage)

            print(f"✅ {len(chunks)} chunks")
            processed += 1

            # Be polite to Wikipedia's servers — 0.5s delay between requests
            time.sleep(0.5)

    # Save to JSON
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in all_passages], f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"✅ Corpus built: {len(all_passages)} passages")
    print(f"   from {processed} articles")
    print(f"   saved to {output_path}")
    print(f"{'=' * 50}\n")

    return all_passages


def load_corpus(path: str = "data/corpus.json") -> list[Passage]:
    """Load a previously built corpus from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Passage(**p) for p in data]


if __name__ == "__main__":
    build_corpus()