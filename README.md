# Cross-Modal Search Demo

Gemini Embedding 2 (`gemini-embedding-2-preview`) を使った、テキストと画像のクロスモーダル検索デモアプリ。BwAI 2026 向け。

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
