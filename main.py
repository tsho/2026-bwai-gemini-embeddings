import os
import pickle
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DB_PATH = Path("db.pkl")

app = FastAPI()


def load_db() -> list[dict]:
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)  # noqa: S301
    return []


def save_db():
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


# In-memory vector DB (loaded from disk)
db: list[dict] = load_db()

EMBEDDING_MODEL = "gemini-embedding-2-preview"


def get_embedding(contents):
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=contents)
    return response.embeddings[0].values


def get_multimodal_embedding(text, image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    content = types.Content(
        parts=[
            types.Part.from_text(text=text),
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
        ]
    )
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=content)
    return response.embeddings[0].values


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.post("/api/index/text")
async def index_text(text: str = Form()):
    vector = get_embedding(text)
    db.append({"type": "text", "content": text, "vector": vector})
    save_db()
    return {"status": "ok", "count": len(db)}


@app.post("/api/index/image")
async def index_image(file: UploadFile = File()):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    vector = get_embedding(image)
    # Save image to uploads/
    ext = Path(file.filename).suffix or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    db.append({
        "type": "image",
        "content": file.filename,
        "image_url": f"/uploads/{filename}",
        "vector": vector,
    })
    save_db()
    return {"status": "ok", "count": len(db)}


@app.post("/api/search")
async def search(
    query_text: str = Form(default=None),
    query_image: UploadFile = File(default=None),
):
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
    for item in db:
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
async def stats():
    text_count = sum(1 for item in db if item["type"] == "text")
    image_count = sum(1 for item in db if item["type"] == "image")
    return {"text_count": text_count, "image_count": image_count, "total": len(db)}


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
