# Cross-Modal Search Demo

[日本語版 / Japanese](README.md)

A cross-modal text-and-image search demo app built on Gemini Embedding 2 (`gemini-embedding-2-preview`). Created for BwAI 2026.

## Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xYsUFGPLsV-vCqznhTqlwh9pSTi_Yc9W)

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Gemini API key](https://aistudio.google.com/apikey)

### Install

```bash
uv sync
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

## Run

```bash
uv run uvicorn main:app --reload
```

Open http://localhost:8000 in your browser.

## Switching to Vertex AI (avoiding 429s)

The AI Studio API key (default) has a strict **per-minute quota of about 15 RPM**, so bursty usage like `seed.py` can hit `429 RESOURCE_EXHAUSTED`. If you expect to push enough traffic to bump into the quota, switch to **GCP Service Account + Vertex AI** for a much higher ceiling (1500 RPM by default, with room to request more).

### 1. Prepare a project

Create or pick a GCP project in [Cloud Console](https://console.cloud.google.com/projectcreate) and note its **project ID**.

### 2. Enable the Vertex AI API

Open the [Vertex AI API page](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) and click **Enable**.

### 3. Create a service account and issue a JSON key

Go to [Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts) → **Create service account**:

1. Name: `vertex-embedding-user` (or anything you like)
2. Role: grant **Vertex AI User** (`roles/aiplatform.user`)
3. Once created, open the SA → **Keys** tab → **Add key** → **Create new key** → **JSON** → download
4. Store the JSON file somewhere safe outside this repo (e.g. `~/keys/vertex-sa.json`)

> Direct link (project-scoped): `https://console.cloud.google.com/iam-admin/serviceaccounts?project=<PROJECT_ID>`

### 4. Edit `.env`

```bash
USE_VERTEX=true
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/vertex-sa.json
```

When `USE_VERTEX=true` is set, the API key is ignored and the app talks to Vertex AI via ADC. Stop the server with Ctrl-C and restart it with `uv run uvicorn main:app --reload`.

## Switching the vector DB to LanceDB

The default backend is an in-memory `list[dict]` persisted via pickle to `db.pkl`. If you'd rather offload persistence and ANN to a dedicated DB, you can switch to LanceDB:

```bash
# .env
USE_LANCE=true
```

With `USE_LANCE=true`, a [LanceDB](https://lancedb.github.io/lancedb/) (Rust-based embedded vector DB) table is created in a local `lance_db/` directory, and inserts, search, and stats all flow through it. `db.pkl` is left untouched (just unset `USE_LANCE` and restart to go back).

The active backend is reported in the uvicorn startup log:

```
INFO:     Embedding backend: AI Studio (GEMINI_API_KEY)
INFO:     Vector DB: LanceDB (lance_db, 0 items loaded)
```

> Note: switching the backend does **not** migrate data automatically. Right after flipping to LanceDB you start from an empty index, so re-run `seed.py` if you need demo data.

## Seeding demo data

While the server is running, this registers 30 text items + 20 image items (50 total) as demo data:

```bash
uv run python seed.py
```

Images are gradient PNGs generated on the fly with Pillow (Sunset, Ocean, Forest, Cherry Blossom, etc.).

## Usage

### 1. Register data

- **Register text**: type into the text box and click "Register Text"
- **Register image**: choose an image file and click "Register Image"

Each registration stores an embedding vector in the in-memory DB.

### 2. Search

- **Text search**: type a query and click "Text Search"
- **Image search**: pick an image and click "Image Search"

You can search images with a text query, or search text with an image — cross-modal search. The top 3 results by cosine similarity are shown as cards.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/index/text` | POST | Register text (`text` form field) |
| `/api/index/image` | POST | Register an image (`file` upload) |
| `/api/search` | POST | Search (`query_text` or `query_image`) |

## Tech stack

- **Backend**: FastAPI
- **Embedding**: Gemini Embedding 2 (`gemini-embedding-2-preview`, 3072 dimensions)
- **Frontend**: HTML + CSS + vanilla JS
- **Vector DB**: in-memory (list[dict])
