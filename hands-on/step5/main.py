"""STEP 5: 本物のベクトル DB (LanceDB) に置き換える.

STEP3 までは ``_db: list[dict]`` の全件スキャンでコサイン類似度を計算していたが、
件数が増えると線形に遅くなる。本 STEP では LanceDB (Rust 製の埋め込み型ベクトル DB)
に置き換える。

ポイント:

- 検索ロジックの **外形** (embed -> 近傍検索 -> top-k) は STEP3 と同じ。
- 変わるのは内部表現 (``list[dict]`` → ``lancedb.Table``) と
  類似度計算 (Python 側のループ → DB 側の ANN/scan)。
- 永続化も DB が肩代わりするので ``pickle`` ハンドリングは消える。

UI は同梱の ``static/index.html`` を使う (ハンズオン対象外)。

Example:
    $ uv run uvicorn hands-on.step5.main:app --reload
    # ブラウザで http://localhost:8000 を開く
"""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-embedding-2-preview"
_DIM = 3072  # gemini-embedding-2-preview のデフォルト次元

_BASE_DIR = Path(__file__).parent
_UPLOAD_DIR = _BASE_DIR / "uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)
_LANCE_DIR = _BASE_DIR / "lance_db"

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

app = FastAPI()


def get_embedding(content: str | Image.Image) -> list[float]:
    """テキストまたは画像を埋め込みベクトルに変換する.

    Args:
        content: テキスト (``str``) または PIL Image。

    Returns:
        埋め込みベクトル (デフォルトでは 3072 次元)。
    """
    res = _client.models.embed_content(model=_MODEL, contents=content)
    return res.embeddings[0].values


def get_multimodal_embedding(text: str, image: Image.Image) -> list[float]:
    """テキストと画像を1つのマルチモーダル入力として埋め込む.

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
    res = _client.models.embed_content(model=_MODEL, contents=content)
    return res.embeddings[0].values


@app.post("/api/index/text")
async def index_text(text: str = Form()) -> dict[str, Any]:
    """テキストを LanceDB テーブルに追加する.

    Args:
        text: 登録するテキスト (multipart form フィールド)。

    Returns:
        ``status`` と登録後の総アイテム数 ``count`` を含む辞書。
    """
    vector = get_embedding(text)
    _table.add(
        [
            {
                "vector": vector,
                "type": "text",
                "content": text,
                "image_url": None,
            }
        ]
    )
    return {"status": "ok", "count": _table.count_rows()}


@app.post("/api/index/image")
async def index_image(file: UploadFile = File()) -> dict[str, Any]:
    """アップロード画像を LanceDB テーブルに追加する.

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
    _table.add(
        [
            {
                "vector": vector,
                "type": "image",
                "content": file.filename,
                "image_url": f"/uploads/{filename}",
            }
        ]
    )
    return {"status": "ok", "count": _table.count_rows()}


@app.post("/api/search")
async def search(
    query_text: str = Form(default=None),
    query_image: UploadFile = File(default=None),
) -> dict[str, Any]:
    """テキスト・画像・両方のいずれかで LanceDB を検索する.

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

    # LanceDB に投げる: metric="cosine" だと _distance は 1 - cosine_similarity になる。
    # text と image を分けて上位5件ずつ欲しいので、まず両合計で十分な件数を取ってから振り分ける。
    hits = _table.search(query_vector).metric("cosine").limit(20).to_list()

    text_results = []
    image_results = []
    for hit in hits:
        score = 1.0 - hit["_distance"]
        result = {"type": hit["type"], "content": hit["content"], "score": score}
        if hit.get("image_url"):
            result["image_url"] = hit["image_url"]
        if hit["type"] == "image":
            image_results.append(result)
        else:
            text_results.append(result)
    return {
        "text_results": text_results[:5],
        "image_results": image_results[:5],
    }


@app.get("/api/stats")
async def stats() -> dict[str, int]:
    """LanceDB テーブルの統計情報を返す.

    Returns:
        テキスト数・画像数・合計数を含む辞書。
    """
    total = _table.count_rows()
    text_count = _table.count_rows(filter="type = 'text'")
    image_count = _table.count_rows(filter="type = 'image'")
    return {"text_count": text_count, "image_count": image_count, "total": total}


app.mount("/uploads", StaticFiles(directory=_UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=_BASE_DIR / "static"), name="static")


@app.get("/")
async def root() -> FileResponse:
    """ルート (``/``) で同梱の UI HTML を返す.

    Returns:
        ``static/index.html`` の ``FileResponse``。
    """
    return FileResponse(_BASE_DIR / "static" / "index.html")
