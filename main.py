"""Cross-modal search demo backend.

Gemini Embedding 2 (``gemini-embedding-2-preview``) を使って、テキスト・画像・
両方を組み合わせたクエリで検索できる小さな FastAPI アプリ。インデックスは
プロセス内の ``list[dict]``、永続化は ``db.pkl`` (pickle)。
"""

from __future__ import annotations

import logging
import os
import pickle
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
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

_UPLOAD_DIR = Path("uploads")
_UPLOAD_DIR.mkdir(exist_ok=True)

_DB_PATH = Path("db.pkl")

app = FastAPI()


def load_db() -> list[dict[str, Any]]:
    """ベクトルDBをディスクから読み込む.

    Returns:
        永続化されたアイテムのリスト。``db.pkl`` が無ければ空リスト。
    """
    if _DB_PATH.exists():
        with open(_DB_PATH, "rb") as f:
            return pickle.load(f)  # noqa: S301
    return []


def save_db() -> None:
    """現在のベクトルDBを ``db.pkl`` に書き出す."""
    with open(_DB_PATH, "wb") as f:
        pickle.dump(_db, f)


_db: list[dict[str, Any]] = load_db()
_logger.info("Vector DB: list[dict] (in-memory + %s, %d items loaded)", _DB_PATH, len(_db))

_EMBEDDING_MODEL = "gemini-embedding-2-preview"


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


@app.post("/api/index/text")
async def index_text(text: str = Form()) -> dict[str, Any]:
    """テキストをインデックスに登録する.

    Args:
        text: 登録するテキスト (multipart form フィールド)。

    Returns:
        ``status`` と登録後の総アイテム数 ``count`` を含む辞書。
    """
    vector = get_embedding(text)
    _db.append({"type": "text", "content": text, "vector": vector})
    save_db()
    return {"status": "ok", "count": len(_db)}


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
    _db.append(
        {
            "type": "image",
            "content": file.filename,
            "image_url": f"/uploads/{filename}",
            "vector": vector,
        }
    )
    save_db()
    return {"status": "ok", "count": len(_db)}


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

    text_results = []
    image_results = []
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
    return {
        "text_results": text_results[:5],
        "image_results": image_results[:5],
    }


@app.get("/api/stats")
async def stats() -> dict[str, int]:
    """インデックスの統計情報を返す.

    Returns:
        テキスト数・画像数・合計数を含む辞書。
    """
    text_count = sum(1 for item in _db if item["type"] == "text")
    image_count = sum(1 for item in _db if item["type"] == "image")
    return {"text_count": text_count, "image_count": image_count, "total": len(_db)}


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root() -> FileResponse:
    """ルート (``/``) で同梱の UI HTML を返す.

    Returns:
        ``static/index.html`` の ``FileResponse``。
    """
    return FileResponse("static/index.html")
