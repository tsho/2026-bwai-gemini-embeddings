# Cross-Modal Search Demo

Gemini Embedding 2 (`gemini-embedding-2-preview`) を使った、テキストと画像のクロスモーダル検索デモアプリ。BwAI 2026 向け。

## Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xYsUFGPLsV-vCqznhTqlwh9pSTi_Yc9W)

## セットアップ

### 前提条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Gemini API キー](https://aistudio.google.com/apikey)

### インストール

```bash
uv sync
cp .env.example .env
# .env を編集して GEMINI_API_KEY を設定
```

## 起動

```bash
uv run uvicorn main:app --reload
```

ブラウザで http://localhost:8000 にアクセス。

## Vertex AI に切り替える（429 対策）

AI Studio の API キー（既定）は **per-minute クォータが ~15 RPM** と厳しく、`seed.py` のような連投で 429 RESOURCE_EXHAUSTED に当たる可能性があります。そのようなクォータにひっかかるような使い方をする場合、 **GCP の Service Account + Vertex AI** に切り替えることで回避できます（クォータが 1500 RPM など大きく上がります）。

### 1. プロジェクトを用意する

[Cloud Console](https://console.cloud.google.com/projectcreate) で GCP プロジェクトを作る（または既存の物を使う）。**プロジェクト ID** を控える。

### 2. Vertex AI API を有効化する

[Vertex AI API ページ](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) を開いて「有効にする」をクリック。

### 3. Service Account を作って JSON キーを発行する

[Service Accounts 一覧](https://console.cloud.google.com/iam-admin/serviceaccounts) → **サービスアカウントの作成**:

1. 名前: `vertex-embedding-user` (or 任意の名前)
2. ロール: **Vertex AI User** （`roles/aiplatform.user`）を付与
3. 作成後、その SA を開く → **鍵** タブ → **キーを追加** → **新しい鍵を作成** → **JSON** → ダウンロード
4. JSON ファイルをリポジトリ外の安全な場所に保管（例: `~/keys/vertex-sa.json`）

> 直リンク（プロジェクト固定）: `https://console.cloud.google.com/iam-admin/serviceaccounts?project=<PROJECT_ID>`

### 4. `.env` を編集する

```bash
USE_VERTEX=true
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/vertex-sa.json
```

`USE_VERTEX=true` が立っていれば API キーは無視され、Vertex AI 経由（ADC 認証）になる。サーバを Ctrl-C で止めて `uv run uvicorn main:app --reload` で再起動。

## Vector DB を LanceDB に切り替える

既定は in-memory `list[dict]` + pickle (`db.pkl`) ですが、永続化と ANN を専用 DB に肩代わりさせたいときは LanceDB に切り替えられます。

```bash
# .env
USE_LANCE=true
```

`USE_LANCE=true` で起動すると、ローカルの `lance_db/` ディレクトリに [LanceDB](https://lancedb.github.io/lancedb/)（Rust 製の埋め込み型ベクトル DB）のテーブルが作られ、追加・検索・統計はすべてそちら経由になります。`db.pkl` には触れません（戻したいときは `USE_LANCE` を外して再起動するだけ）。

uvicorn の起動ログにどちらを使っているかが表示されます。

```
INFO:     Embedding backend: AI Studio (GEMINI_API_KEY)
INFO:     Vector DB: LanceDB (lance_db, 0 items loaded)
```

> 注意: backend を切り替えてもデータの自動移行はしません。LanceDB に切り替えた直後は空インデックスから始まるので、必要に応じて `seed.py` で再投入してください。

## デモデータ投入

サーバー起動中に以下を実行すると、テキスト30件 + 画像20件 = 計50件のデモデータが登録される。

```bash
uv run python seed.py
```

画像は Pillow でグラデーション画像を自動生成する（Sunset, Ocean, Forest, Cherry Blossom など）。

## 使い方

### 1. データ登録

- **テキスト登録**: テキストを入力して「Register Text」をクリック
- **画像登録**: 画像ファイルを選択して「Register Image」をクリック

登録するとインメモリDBに embedding ベクトルが保存される。

### 2. 検索

- **テキスト検索**: 検索クエリを入力して「Text Search」をクリック
- **画像検索**: 画像ファイルを選択して「Image Search」をクリック

テキストで画像を検索したり、画像でテキストを検索するクロスモーダル検索が可能。結果はコサイン類似度の上位3件がカード形式で表示される。

## API

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/api/index/text` | POST | テキストを登録 (`text` フォーム) |
| `/api/index/image` | POST | 画像を登録 (`file` アップロード) |
| `/api/search` | POST | 検索 (`query_text` or `query_image`) |

## 技術構成

- **バックエンド**: FastAPI
- **Embedding**: Gemini Embedding 2 (`gemini-embedding-2-preview`, 3072次元)
- **フロントエンド**: HTML + CSS + vanilla JS
- **ベクトルDB**: インメモリ (list[dict])
