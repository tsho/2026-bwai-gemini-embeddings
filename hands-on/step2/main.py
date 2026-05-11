"""STEP 2: クロスモーダル化.

STEP1 の ``get_embedding()`` に PIL Image を渡すだけで画像もベクトル化できる、
というのが核心。テキストと画像が同じ 3072 次元空間に乗るため、
クロスモーダル検索 (テキスト -> 画像、画像 -> テキスト) と、
**マルチモーダル検索** (テキスト + 画像を1つのクエリとして扱う) の
両方が可能になる。

Example:
    $ uv run python hands-on/step2/main.py
"""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw

load_dotenv()
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-embedding-2-preview"

_TEXTS = [
    "東京タワーは1958年に完成した電波塔で、高さは333メートルです",
    "富士山は日本最高峰の山で、標高3776メートルです",
    "桜は日本の春を象徴する花で、3月から4月にかけて咲きます",
    "A tropical beach with turquoise water and white sand",
    "A snowy mountain peak with clear blue sky",
    "A field of sunflowers stretching to the horizon",
    "A cozy library filled with old leather-bound books",
    "A campfire under a starry night sky in the forest",
]

_IMAGES = [
    ("sunset", "#FF6B35", "#FFC300"),
    ("ocean", "#006994", "#40E0D0"),
    ("forest", "#228B22", "#90EE90"),
    ("cherry_blossom", "#FFB7C5", "#FF69B4"),
    ("snow", "#E8E8E8", "#FFFFFF"),
    ("night_sky", "#0C1445", "#4169E1"),
]


def make_gradient(color1: str, color2: str) -> Image.Image:
    """2色の縦方向グラデーション画像を生成する.

    Args:
        color1: 上端の色 (CSSカラー文字列、例: ``"#FF6B35"``)。
        color2: 下端の色 (CSSカラー文字列)。

    Returns:
        256x256 の RGB 画像。
    """
    w, h = 256, 256
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    r1, g1, b1 = Image.new("RGB", (1, 1), color1).getpixel((0, 0))
    r2, g2, b2 = Image.new("RGB", (1, 1), color2).getpixel((0, 0))
    for y in range(h):
        t = y / h
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def get_embedding(content: str | Image.Image) -> list[float]:
    """テキストまたは画像を埋め込みベクトルに変換する.

    Gemini Embedding 2 はテキストと画像を同じ空間に埋め込むため、
    呼び出し側はどちらでも同じ関数で扱える。

    Args:
        content: テキスト (``str``) または PIL Image。

    Returns:
        埋め込みベクトル (デフォルトでは 3072 次元)。
    """
    res = _client.models.embed_content(model=_MODEL, contents=content)
    return res.embeddings[0].values


def get_multimodal_embedding(text: str, image: Image.Image) -> list[float]:
    """テキストと画像を1つのマルチモーダル入力として埋め込む.

    ``types.Content`` に複数の ``Part`` (テキスト + 画像) をまとめて渡すことで、
    両モダリティの情報を反映した単一の埋め込みベクトルが得られる。
    cross-modal (どちらか一方) ではなく multimodal (両方を組み合わせ) クエリ。

    Args:
        text: クエリの一部としてのテキスト。
        image: クエリの一部としての画像。

    Returns:
        テキスト + 画像の組み合わせを表現する単一の埋め込みベクトル。
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    content = types.Content(
        parts=[
            types.Part.from_text(text=text),
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
        ]
    )
    res = _client.models.embed_content(model=_MODEL, contents=content)
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
    query: str | Image.Image,
    db: list[dict[str, Any]],
    k: int = 3,
) -> list[tuple[float, dict[str, Any]]]:
    """クエリに対するインデックス内の上位 k 件を返す.

    Args:
        query: テキストまたは画像のクエリ。
        db: ``{"type", "content", "vector"}`` を持つアイテムのリスト。
        k: 返す件数の上限。

    Returns:
        ``(score, item)`` のタプルをスコア降順で並べたリスト (最大 k 件)。
    """
    qv = get_embedding(query)
    scored = sorted(
        ((cosine_similarity(qv, item["vector"]), item) for item in db),
        key=lambda x: x[0],
        reverse=True,
    )
    return scored[:k]


def main() -> None:
    """テキスト + 画像インデックスを構築し、クロスモーダル検索を実演する."""
    print("Building index (text + image)...")
    db: list[dict[str, Any]] = []
    for text in _TEXTS:
        db.append({"type": "text", "content": text, "vector": get_embedding(text)})
    for name, c1, c2 in _IMAGES:
        img = make_gradient(c1, c2)
        db.append({"type": "image", "content": name, "vector": get_embedding(img)})
    n_text = sum(1 for x in db if x["type"] == "text")
    n_image = sum(1 for x in db if x["type"] == "image")
    print(f"Indexed {len(db)} items ({n_text} texts, {n_image} images).\n")

    text_queries = [
        "sunset over the ocean",
        "夜空に輝く星",
        "日本の春",
    ]
    for q in text_queries:
        print(f"Query (text): {q}")
        for score, item in search(q, db):
            print(f"  {score:.4f}  [{item['type']:5}] {item['content']}")
        print()

    query_img = make_gradient("#FF6B35", "#FFC300")
    print("Query (image): orange-yellow gradient (sunset-like)")
    for score, item in search(query_img, db):
        print(f"  {score:.4f}  [{item['type']:5}] {item['content']}")
    print()

    # Multimodal query: text + image を1つのクエリとして組み合わせる
    mm_text = "日本"
    mm_image = make_gradient("#FFB7C5", "#FF69B4")  # cherry-blossom-like pink
    print(f"Query (multimodal): text={mm_text!r} + image=pink gradient (cherry-blossom-like)")
    mm_vec = get_multimodal_embedding(mm_text, mm_image)
    scored = sorted(
        ((cosine_similarity(mm_vec, item["vector"]), item) for item in db),
        key=lambda x: x[0],
        reverse=True,
    )
    for score, item in scored[:3]:
        print(f"  {score:.4f}  [{item['type']:5}] {item['content']}")


if __name__ == "__main__":
    main()
