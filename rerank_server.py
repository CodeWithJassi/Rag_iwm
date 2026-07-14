"""Minimal reranker server for the cluster — single /rerank endpoint,
same request/response format as TEI so _rerank_remote() works unchanged.

Usage:
  pip install flask sentence-transformers
  python rerank_server.py

Env vars:
  RERANK_MODEL   — model id (default: BAAI/bge-reranker-v2-m3)
  RERANK_PORT    — listen port (default: 8080)
  RERANK_HOST    — listen host (default: 0.0.0.0)
"""
import os

from flask import Flask, jsonify, request
from sentence_transformers import CrossEncoder

MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
PORT = int(os.getenv("RERANK_PORT", "8080"))
HOST = os.getenv("RERANK_HOST", "0.0.0.0")

app = Flask(__name__)

print(f"Loading reranker model: {MODEL} ...")
reranker = CrossEncoder(MODEL)
print("Model loaded, server ready.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/rerank", methods=["POST"])
def rerank():
    data = request.get_json()
    query = data["query"]
    texts = data["texts"]
    pairs = [[query, t] for t in texts]
    scores = reranker.predict(pairs)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if not isinstance(scores, list):
        scores = [scores]
    return jsonify([
        {"index": i, "score": round(float(s), 4)}
        for i, s in enumerate(scores)
    ])


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
