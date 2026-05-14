"""Cross-modal search demo backend.

Gemini Embedding 2 (``gemini-embedding-2-preview``) を使って、テキスト・画像・
両方を組み合わせたクエリで検索できる小さな FastAPI アプリ。

embedding API と vector store はどちらも環境変数で切り替えられる:

- ``USE_VERTEX=true``  → AI Studio から Vertex AI (Service Account) に切替
- ``USE_LANCE=true``   → in-memory ``list[dict]`` + pickle から LanceDB に切替

詳細は README の対応セクションを参照。
"""

from __future__ import annotations

import logging
import os
import pickle
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

# uvicorn が設定済みのロガーに乗っかると、起動メッセージと同じスタイルで出る。
_logger = logging.getLogger("uvicorn.error")

# --- Embedding backend (AI Studio / Vertex AI) ---
_USE_VERTEX = os.environ.get("USE_VERTEX", "").lower() in {"1", "true", "yes"}

if _USE_VERTEX:
    # Service Account 経由で Vertex AI を使う。クォータが緩いので、複数人の
    # 同時実行や seed.py の連投にも耐えやすい。
    # 認証は GOOGLE_APPLICATION_CREDENTIALS が指す JSON キー (ADC) で行う。
    _project = os.environ["GOOGLE_CLOUD_PROJECT"]
    _location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    _client = genai.Client(vertexai=True, project=_project, location=_location)
    _logger.info("Embedding backend: Vertex AI (project=%s, location=%s)", _project, _location)
else:
    _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    _logger.info("Embedding backend: AI Studio (GEMINI_API_KEY)")

_EMBEDDING_MODEL = "gemini-embedding-2-preview"
_DIM = 3072  # gemini-embedding-2-preview のデフォルト次元

# --- Vector store backend (list[dict] / LanceDB) ---
_USE_LANCE = os.environ.get("USE_LANCE", "").lower() in {"1", "true", "yes"}

_UPLOAD_DIR = Path("uploads")
_UPLOAD_DIR.mkdir(exist_ok=True)

if _USE_LANCE:
    _LANCE_DIR = Path("lance_db")
    _SCHEMA = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), _DIM)),
            pa.field("type", pa.string()),
            pa.field("content", pa.string()),
            pa.field("image_url", pa.string()),
        ]
    )
    _lance = lancedb.connect(str(_LANCE_DIR))
    _table = _lance.create_table("items", schema=_SCHEMA, exist_ok=True)
    _logger.info("Vector DB: LanceDB (%s, %d items loaded)", _LANCE_DIR, _table.count_rows())
else:
    _DB_PATH = Path("db.pkl")

    def _load_db() -> list[dict[str, Any]]:
        """``db.pkl`` からベクトル DB をロードする (無ければ空リスト)."""
        if _DB_PATH.exists():
            with open(_DB_PATH, "rb") as f:
                return pickle.load(f)  # noqa: S301
        return []

    def _save_db() -> None:
        """現在のベクトル DB を ``db.pkl`` に書き出す."""
        with open(_DB_PATH, "wb") as f:
            pickle.dump(_db, f)

    _db: list[dict[str, Any]] = _load_db()
    _logger.info("Vector DB: list[dict] (%s, %d items loaded)", _DB_PATH, len(_db))


app = FastAPI()


def get_embedding(contents: str | Image.Image) -> list[float]:
    """テキストまたは画像を埋め込みベクトルに変換する.

    Args:
        contents: テキスト (``str``) または PIL Image。

    Returns:
        埋め込みベクトル (デフォルトでは 3072 次元)。
    """
    response = _client.models.embed_content(model=_EMBEDDING_MODEL, contents=contents)
    return response.embeddings[0].values


def get_multimodal_embedding(text: str, image: Image.Image) -> list[float]:
    """テキストと画像を1つのマルチモーダル入力として埋め込む.

    ``types.Content`` に複数の ``Part`` をまとめて渡すことで、テキストと画像
    両方の情報を反映した単一の埋め込みベクトルが得られる。

    Args:
        text: クエリの一部としてのテキスト。
        image: クエリの一部としての画像。

    Returns:
        テキスト + 画像の組み合わせを表現する単一の埋め込みベクトル。
    """
    buf = BytesIO()
    image.save(buf, format="PNG")
    content = types.Content(
        parts=[
            types.Part.from_text(text=text),
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
        ]
    )
    response = _client.models.embed_content(model=_EMBEDDING_MODEL, contents=content)
    return response.embeddings[0].values


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


def _store_add(item: dict[str, Any]) -> int:
    """選択中のバックエンドにアイテムを追加し、追加後の総件数を返す.

    Args:
        item: ``type`` / ``content`` / ``vector`` (任意で ``image_url``) を持つ辞書。

    Returns:
        追加後の総アイテム数。
    """
    if _USE_LANCE:
        _table.add(
            [
                {
                    "vector": item["vector"],
                    "type": item["type"],
                    "content": item["content"],
                    "image_url": item.get("image_url"),
                }
            ]
        )
        return _table.count_rows()
    _db.append(item)
    _save_db()
    return len(_db)


def _store_search(
    query_vector: list[float], k: int = 5
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """クエリベクトルに対する上位 k 件を type 別に分けて返す.

    Args:
        query_vector: 検索クエリの埋め込みベクトル。
        k: テキスト・画像それぞれで返す件数。

    Returns:
        ``(text_results, image_results)`` のタプル。スコア降順で各最大 k 件。
    """
    text_results: list[dict[str, Any]] = []
    image_results: list[dict[str, Any]] = []
    if _USE_LANCE:
        # type 別に k 件欲しいので、ANN 側では多めに取ってから振り分ける。
        hits = _table.search(query_vector).metric("cosine").limit(k * 4).to_list()
        for hit in hits:
            score = 1.0 - hit["_distance"]
            result = {"type": hit["type"], "content": hit["content"], "score": score}
            if hit.get("image_url"):
                result["image_url"] = hit["image_url"]
            if hit["type"] == "image":
                image_results.append(result)
            else:
                text_results.append(result)
        # LanceDB は距離昇順で返すので、type 別にしただけで降順ソート済み。
    else:
        for item in _db:
            score = cosine_similarity(query_vector, item["vector"])
            result = {"type": item["type"], "content": item["content"], "score": score}
            if item.get("image_url"):
                result["image_url"] = item["image_url"]
            if item["type"] == "image":
                image_results.append(result)
            else:
                text_results.append(result)
        text_results.sort(key=lambda x: x["score"], reverse=True)
        image_results.sort(key=lambda x: x["score"], reverse=True)
    return text_results[:k], image_results[:k]


def _store_stats() -> tuple[int, int, int]:
    """選択中のバックエンドの統計を返す.

    Returns:
        ``(text_count, image_count, total)`` のタプル。
    """
    if _USE_LANCE:
        total = _table.count_rows()
        text_count = _table.count_rows(filter="type = 'text'")
        image_count = _table.count_rows(filter="type = 'image'")
    else:
        text_count = sum(1 for x in _db if x["type"] == "text")
        image_count = sum(1 for x in _db if x["type"] == "image")
        total = len(_db)
    return text_count, image_count, total


@app.post("/api/index/text")
async def index_text(text: str = Form()) -> dict[str, Any]:
    """テキストをインデックスに登録する.

    Args:
        text: 登録するテキスト (multipart form フィールド)。

    Returns:
        ``status`` と登録後の総アイテム数 ``count`` を含む辞書。
    """
    vector = get_embedding(text)
    count = _store_add({"type": "text", "content": text, "vector": vector})
    return {"status": "ok", "count": count}


@app.post("/api/index/image")
async def index_image(file: UploadFile = File()) -> dict[str, Any]:
    """アップロード画像をインデックスに登録する.

    画像本体は ``uploads/<uuid>.<ext>`` に保存し、検索結果から参照できるように
    ``image_url`` フィールドに公開パスを記録する。

    Args:
        file: アップロードされた画像ファイル。

    Returns:
        ``status`` と登録後の総アイテム数 ``count`` を含む辞書。
    """
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    vector = get_embedding(image)
    ext = Path(file.filename).suffix or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = _UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    count = _store_add(
        {
            "type": "image",
            "content": file.filename,
            "image_url": f"/uploads/{filename}",
            "vector": vector,
        }
    )
    return {"status": "ok", "count": count}


@app.post("/api/search")
async def search(
    query_text: str = Form(default=None),
    query_image: UploadFile = File(default=None),
) -> dict[str, Any]:
    """テキスト・画像・両方のいずれかでインデックスを検索する.

    - 両方指定: ``text + image`` を1つのマルチモーダルクエリにまとめる。
    - 片方のみ: クロスモーダル検索 (テキスト or 画像)。

    Args:
        query_text: テキストクエリ (任意)。
        query_image: 画像クエリ (任意)。

    Returns:
        テキスト結果と画像結果のそれぞれ上位5件を含む辞書、
        または両方未指定の場合はエラー辞書。
    """
    if query_text and query_image:
        image_bytes = await query_image.read()
        image = Image.open(BytesIO(image_bytes))
        query_vector = get_multimodal_embedding(query_text, image)
    elif query_text:
        query_vector = get_embedding(query_text)
    elif query_image:
        image_bytes = await query_image.read()
        image = Image.open(BytesIO(image_bytes))
        query_vector = get_embedding(image)
    else:
        return {"error": "query_text or query_image is required"}

    text_results, image_results = _store_search(query_vector, k=5)
    return {"text_results": text_results, "image_results": image_results}


@app.get("/api/stats")
async def stats() -> dict[str, int]:
    """インデックスの統計情報を返す.

    Returns:
        テキスト数・画像数・合計数を含む辞書。
    """
    text_count, image_count, total = _store_stats()
    return {"text_count": text_count, "image_count": image_count, "total": total}


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root() -> FileResponse:
    """ルート (``/``) で同梱の UI HTML を返す.

    Returns:
        ``static/index.html`` の ``FileResponse``。
    """
    return FileResponse("static/index.html")
