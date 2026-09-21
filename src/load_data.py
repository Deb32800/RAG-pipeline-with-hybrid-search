"""Load the SciFact dataset (corpus, queries, answer key) into plain Python dicts."""
import csv
import json
from pathlib import Path

# .../RAG/src/load_data.py -> .../RAG/data/scifact
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"


def load_corpus() -> dict[str, dict]:
    """Return {doc_id: {"title": ..., "text": ...}} for all 5,183 abstracts."""
    corpus = {}
    with open(DATA_DIR / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"title": doc["title"], "text": doc["text"]}
    return corpus


def load_queries() -> dict[str, str]:
    """Return {query_id: claim_text} for all 1,109 claims (train and test)."""
    queries = {}
    with open(DATA_DIR / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(split: str = "test") -> dict[str, set[str]]:
    """Return the answer key {query_id: {correct doc ids}} for 'train' or 'test'."""
    qrels = {}
    with open(DATA_DIR / "qrels" / f"{split}.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return qrels


def doc_text(doc: dict) -> str:
    """The text we search over: title and abstract joined. One doc = one chunk."""
    return f"{doc['title']}. {doc['text']}"


if __name__ == "__main__":
    corpus, queries, qrels = load_corpus(), load_queries(), load_qrels("test")
    print(f"{len(corpus)} docs | {len(queries)} queries | {len(qrels)} test queries")
