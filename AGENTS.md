# AGENTS.md

This file provides guidance to Cortex Code, Claude Code (claude.ai/code) when working with code in this repository.

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

## Multimodal vs Cross-modal クエリ

「クロスモーダル」と「マルチモーダル」は区別する。

- **クロスモーダル**: テキスト *か* 画像 *のどちらか* をクエリにする。`embed_content(contents=text)` / `embed_content(contents=image)` でそれぞれ別々に埋め込む。
- **マルチモーダル（単一クエリ）**: テキスト *と* 画像 *の両方* を1つのクエリにまとめる。両方の情報を反映した **単一の埋め込みベクトル** を返す。

マルチモーダルクエリの組み立て方:

```python
from google.genai import types

content = types.Content(parts=[
    types.Part.from_text(text="日本"),
    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
])
res = client.models.embed_content(model="gemini-embedding-2-preview", contents=content)
vector = res.embeddings[0].values  # ← テキスト+画像を反映した単一ベクトル
```

**罠**: `list[Part]` をそのまま `contents=` に渡すと **バッチ扱い**（各 Part ごとに別々の embedding が返る）になる。**1つの埋め込みにまとめたいときは必ず `types.Content` でラップする** こと。SDK の `embed_content` の型シグネチャ上は両方受け付けるので静的には気付けない。

API 側のハンドラ実装では、`query_text` と `query_image` の両方が来たときだけマルチモーダル経路に分岐し、片方なら従来通りクロスモーダルにする（`main.py:/api/search` 参照）。UI 側は画像選択で自動検索しないようにして、検索ボタン押下で text + image を同時送信できるようにする必要がある。

## 発展トピック（`task_type` / `output_dimensionality`）

`embed_content` の `config=types.EmbedContentConfig(...)` で、検索品質と効率を改善できる。

- `task_type`: ドキュメント側は `"RETRIEVAL_DOCUMENT"`、クエリ側は `"RETRIEVAL_QUERY"` を指定すると検索向けに最適化された埋め込みになる。
- `output_dimensionality`: 3072 → 768 など低次元に圧縮。メモリ・ストレージ・類似度計算コストはすべて次元数に比例するので、件数が増えるほど効く。

`hands-on/step4/main.py` に baseline / task_type 指定 / 次元削減 の3パターン比較デモがある。

## コード規約 (ruff + Google Style)

`pyproject.toml` で次の方針を採用している。

- **docstring**: Google convention（`[tool.ruff.lint.pydocstyle] convention = "google"`）。`D203` と `D213` は互換性がない組み合わせなので ignore。
- **FastAPI の bugbear 回避**: `File()` / `Form()` / `Depends()` などは引数デフォルトで呼ぶ前提のため、`flake8-bugbear.extend-immutable-calls` に列挙して B008 を回避する。
- **命名 (Google Style §3.16.4)**: モジュール内部に閉じている定数・変数は **`_` プレフィックス** を付ける（例: `_client`, `_MODEL`, `_db`, `_DOCS`）。`app`（uvicorn が `main:app` で参照する公開シンボル）はプレフィックスなし。関数は内部用でもプレフィックスなしで運用している（教育的読みやすさ優先）。
- **format**: `ruff format` も CI で `--check` する。
- **CI**: `.github/workflows/lint.yml` が PR と main への push で `ruff check` + `ruff format --check` を実行。`astral-sh/setup-uv@v5` でキャッシュ、`uv sync --frozen` で再現性確保。

## ハンズオン構成 (`hands-on/`)

`hands-on/` 配下に step1/ ～ step4/ を置く形式。**前のSTEPに最小の差分を加える**ことで概念を1つずつ積み上げる構成にしている。

- **step1**: 最小スクリプト（embed → cosine → top-k）。インデックス・永続化・フレームワークなし。
- **step2**: 同じ `get_embedding()` に画像を渡してクロスモーダル化＋マルチモーダルクエリ。
- **step3**: FastAPI 化（永続化あり）。`main.py` の教育用コピー。**UI は配布済み** として `step3/static/index.html` に置き、ハンズオン対象外。
- **step4**: `task_type` / `output_dimensionality` 比較デモ。発展課題。

各 step は完成品として置いてあり、ハンズオン参加者が詰まったときの答え合わせや差分確認に使う想定。

## Workflow Conventions

- **コミット粒度**: 論理的に独立した変更は別コミットに分ける（例: ruff 設定の追加と機能追加を混ぜない、CLAUDE.md 整理と feature 追加を混ぜない）。
- **大きなチャンクごとに commit + push してから次のタスクに進む** ことをユーザが好む。
- **教育系コンテンツの設計**: 実装前に「どう分けるか」をユーザと合意してから着手する。
- **`AGENTS.md` が真実の源**: `CLAUDE.md` は `@AGENTS.md` で参照しているだけ。プロジェクト文書はここに集約する。
