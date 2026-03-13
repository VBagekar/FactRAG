from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from factrag.bm25_retriever import RetrievedPassage


READER_MODEL = "deepset/roberta-base-squad2"


@dataclass
class Answer:
    text: str
    score: float
    source_title: str
    source_passage: str
    source_url: str


class Reader:
    def __init__(self):
        print(f"Loading reader model: {READER_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL)
        self.model = AutoModelForQuestionAnswering.from_pretrained(READER_MODEL)
        self.model.eval()
        print("✅ Reader model loaded")

    def _predict(self, question: str, context: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)

        # end must come after start
        if end_idx < start_idx:
            return "", 0.0

        tokens = inputs["input_ids"][0][start_idx: end_idx + 1]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

        # Confidence = softmax probability of start * end positions
        start_prob = torch.softmax(outputs.start_logits, dim=1)[0][start_idx].item()
        end_prob = torch.softmax(outputs.end_logits, dim=1)[0][end_idx].item()
        score = round(start_prob * end_prob, 4)

        return answer, score

    def extract_answer(self,
                       question: str,
                       passages: list[RetrievedPassage],
                       top_k: int = 3) -> Optional[Answer]:
        if not passages:
            return None

        best_answer = None
        best_score = -1

        for retrieved in passages[:top_k]:
            answer_text, score = self._predict(question, retrieved.passage.text)

            if score > best_score and answer_text:
                best_score = score
                best_answer = Answer(
                    text=answer_text,
                    score=score,
                    source_title=retrieved.passage.title,
                    source_passage=retrieved.passage.text[:200] + "...",
                    source_url=retrieved.passage.source_url,
                )

        return best_answer