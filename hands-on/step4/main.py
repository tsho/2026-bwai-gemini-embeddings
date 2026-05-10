"""STEP 4: 実用化に向けた発展.

``embed_content`` の ``config`` パラメータを使って、検索品質と効率を改善する
2つの設定を比較するデモ。

  - ``task_type``: ドキュメント側は ``RETRIEVAL_DOCUMENT``、クエリ側は
    ``RETRIEVAL_QUERY`` を指定すると検索向けに最適化された埋め込みになる。
  - ``output_dimensionality``: 次元数を下げてストレージ・計算コストを削減。

3パターンを並べて、同じクエリでスコアと順位がどう変わるかを観察する。

Example:
    $ uv run python hands-on/step4/main.py
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-embedding-2-preview"

DOCS = [
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

QUERY = "日本の歴史的な観光地"


def embed(content: str, config: types.EmbedContentConfig | None = None) -> list[float]:
    """設定付きで埋め込みベクトルを取得する.

    Args:
        content: 埋め込み対象のテキスト。
        config: ``task_type`` や ``output_dimensionality`` を含むオプション設定。

    Returns:
        埋め込みベクトル。``config.output_dimensionality`` が指定されていれば
        その次元数、未指定ならデフォルトの 3072 次元。
    """
    res = client.models.embed_content(model=MODEL, contents=content, config=config)
    return res.embeddings[0].values


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """2つのベクトル間のコサイン類似度を計算する.

    Args:
        a: 1つ目のベクトル。
        b: 2つ目のベクトル。

    Returns:
        -1.0 から 1.0 の範囲のコサイン類似度。
    """
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def search(
    query_vec: list[float],
    db: list[dict[str, Any]],
    k: int = 3,
) -> list[tuple[float, dict[str, Any]]]:
    """クエリベクトルに対する上位 k 件を返す.

    Args:
        query_vec: クエリの埋め込みベクトル。
        db: ``{"content", "vector"}`` を持つアイテムのリスト。
        k: 返す件数の上限。

    Returns:
        ``(score, item)`` のタプルをスコア降順で並べたリスト (最大 k 件)。
    """
    scored = sorted(
        ((cosine_similarity(query_vec, item["vector"]), item) for item in db),
        key=lambda x: x[0],
        reverse=True,
    )
    return scored[:k]


def run_experiment(
    name: str,
    doc_config: types.EmbedContentConfig | None,
    query_config: types.EmbedContentConfig | None,
) -> None:
    """指定された埋め込み設定で検索を実行し、上位3件を出力する.

    Args:
        name: 実験名 (出力ヘッダに使う)。
        doc_config: ドキュメント側に渡す埋め込み設定。
        query_config: クエリ側に渡す埋め込み設定。
    """
    print(f"=== {name} ===")
    db = [{"content": text, "vector": embed(text, doc_config)} for text in DOCS]
    qv = embed(QUERY, query_config)
    print(f"  vector dim: {len(qv)}")
    print(f"  query: {QUERY}")
    for score, item in search(qv, db):
        print(f"    {score:.4f}  {item['content']}")
    print()


def main() -> None:
    """3つの埋め込み設定で同じクエリを実行し、結果を並べて表示する."""
    # (A) ベースライン: configなし (STEP1〜3と同じ呼び出し)
    run_experiment("A: baseline (no config)", None, None)

    # (B) task_type を指定: ドキュメントとクエリで別の task_type を使う
    run_experiment(
        "B: with task_type (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)",
        types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    # (C) 次元削減: 3072 -> 768 次元に圧縮
    run_experiment(
        "C: task_type + output_dimensionality=768",
        types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
        types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
    )

    print("観察ポイント:")
    print("  - A vs B: 順位やスコア差が変わる (検索向け最適化の効果)")
    print("  - B vs C: 次元が 1/4 でも順位は概ね保たれる (次元削減の許容範囲)")


if __name__ == "__main__":
    main()
