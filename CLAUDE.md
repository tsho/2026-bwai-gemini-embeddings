# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gemini Embedding 2 (`gemini-embedding-2-preview`) を使ったクロスモーダル検索デモアプリ。BwAI 2026 向け。
テキストと画像を同じベクトル空間に埋め込み、テキスト→画像、画像→テキストの横断検索ができる。

## Tech Stack

- Python 3.12
- FastAPI (バックエンド)
- google-genai (Gemini Embedding API クライアント)
- numpy (コサイン類似度計算)
- Pillow (画像処理)
- uv (パッケージ管理)

## Project Structure

- `main.py` — FastAPI アプリ本体（API エンドポイント、embedding取得、検索ロジック）
- `seed.py` — デモデータ投入スクリプト（テキスト30件 + 画像20件）
- `static/index.html` — フロントエンド（Google検索風UI、2カラム結果表示）
- `db.pkl` — ベクトルDB永続化ファイル（pickle、gitignore対象）
- `uploads/` — アップロード画像保存ディレクトリ（gitignore対象）
- `.env` — `GEMINI_API_KEY` を記載（gitignore対象、`.env.example` を参照）

## Commands

```bash
uv sync                              # 依存インストール
uv run uvicorn main:app --reload     # サーバー起動 (http://localhost:8000)
uv run python seed.py                # デモデータ投入（サーバー起動中に実行）
uv run ruff check .                  # lint
```

## Key Design Decisions

- **インメモリDB + pickle永続化**: デモ用途のため簡易実装。サーバー起動時に `db.pkl` から復元。
- **検索結果はテキスト/画像を分離して返す**: テキスト同士の類似度が高くなりがちなため、`text_results` と `image_results` を別々に上位5件ずつ返す設計にした。
- **画像は `uploads/` に保存**: embedding取得後に画像ファイルを保存し、検索結果で画像を表示できるようにした。

## Gemini Embedding API Usage

```python
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# テキスト embedding
response = client.models.embed_content(model="gemini-embedding-2-preview", contents="テキスト")

# 画像 embedding（PIL.Image オブジェクトをそのまま渡せる）
from PIL import Image
image = Image.open("photo.png")
response = client.models.embed_content(model="gemini-embedding-2-preview", contents=image)

# 結果: response.embeddings[0].values → 3072次元のベクトル (list[float])
```

## Learnings

- `gemini-embedding-2-preview` はテキストと画像の両方を同じ3072次元ベクトル空間に埋め込める
- テキスト→テキストの類似度はテキスト→画像より高くなりやすい。クロスモーダル検索デモでは結果をタイプ別に分けて表示するのが有効
- `google-genai` の `embed_content()` は PIL.Image をそのまま渡せるので画像の前処理は不要
- seed スクリプトで Pillow のグラデーション画像を生成すれば、実画像なしでもデモが動作する
- インメモリDBはサーバー再起動でデータが消えるため、デモ用途でも pickle 等での永続化が必要
