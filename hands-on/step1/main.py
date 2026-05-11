"""STEP 1: 最小のテキスト検索エンジン.

ベクトル検索の骨格 (embed -> cosine similarity -> top-k) を体感するための
最小スクリプト。インデックス・永続化・Webフレームワークなどは一切使わない。

Example:
    $ uv run python hands-on/step1/main.py
"""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-embedding-2-preview"

_DOCS = [
    "東京タワーは1958年に完成した電波塔で、高さは333メートルです",
    "富士山は日本最高峰の山で、標高3776メートルです",
    "桜は日本の春を象徴する花で、3月から4月にかけて咲きます",
    "新幹線は時速300キロ以上で走る日本の高速鉄道です",
    "寿司は酢飯の上に新鮮な魚介類をのせた日本の伝統料理です",
    "京都には数多くの寺社仏閣があり、世界遺産にも登録されています",
    "A golden retriever playing fetch in a sunny park",
    "A tropical beach with turquoise water and white sand",
    "A snowy mountain peak with clear blue sky",
    "A cup of hot coffee with latte art on a wooden table",
]


def get_embedding(content: str) -> list[float]:
    """テキストを埋め込みベクトルに変換する.

    Args:
        content: ベクトル化したいテキスト。

    Returns:
        埋め込みベクトル (Gemini Embedding 2 のデフォルトでは 3072 次元)。
    """
    res = _client.models.embed_content(model=_MODEL, contents=content)
    return res.embeddings[0].values


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """2つのベクトル間のコサイン類似度を計算する.

    Args:
        a: 1つ目のベクトル。
        b: 2つ目のベクトル。

    Returns:
        -1.0 から 1.0 の範囲のコサイン類似度。値が大きいほど類似度が高い。
    """
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def main() -> None:
    """インデックスを構築し、2つのクエリで上位3件を表示する."""
    print("Building index...")
    db = [{"content": text, "vector": get_embedding(text)} for text in _DOCS]
    print(f"Indexed {len(db)} documents.\n")

    queries = ["日本の歴史的な建造物", "warm drink in a cafe"]
    for query in queries:
        qv = get_embedding(query)
        scored = sorted(
            ((cosine_similarity(qv, item["vector"]), item) for item in db),
            key=lambda x: x[0],
            reverse=True,
        )
        print(f"Query: {query}")
        for score, item in scored[:3]:
            print(f"  {score:.4f}  {item['content']}")
        print()


if __name__ == "__main__":
    main()
