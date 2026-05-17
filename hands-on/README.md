# Gemini Embedding ハンズオン: クロスモーダル検索システムを作る

[English version is below ↓](#gemini-embedding-hands-on-build-a-cross-modal-search-system-english)

Gemini Embedding API の基本（`embed_content` でベクトルが得られる）を一度でも触ったことがある人向けの、検索システム構築ハンズオンです。もし触っていないひとがいましたら本Github repoのrootにあるREADME.mdにはじめて触る方むけのColabが用意されていますのでそちらをご覧ください。

ゴールは **「テキストでも画像でもクエリにできるクロスモーダル検索システム」** を、概念を1つずつ積み上げながら自分の手で動かすことです。

---

## 全体像

| STEP | 何を作るか | 学ぶこと |
|---|---|---|
| 1 | 最小のテキスト検索エンジン（1ファイル・30行） | 検索＝ベクトル空間の近傍探索という骨格 |
| 2 | クロスモーダル化（同じ関数に画像を渡す） | テキストと画像が同じベクトル空間に乗る感覚 |
| 3 | Web API + UI 化（FastAPI + 永続化） | 検索システムを"使える形"に組み立てる実装パターン |
| 4 | 実用化に向けた発展（任意） | プロダクションへの橋渡し |
| 5 | 本物のベクトル DB に置き換え（LanceDB） | `list[dict]` → 専用 DB への移行が "差し替え" で済む感覚 |

各STEPは前のSTEPに**最小の差分**を加えていく形になっています。STEP1〜2 は1つのスクリプトで完結し、STEP3 で初めて Web フレームワークが登場します。

## ディレクトリ構成

各 STEP の完成品が `step1/` 〜 `step4/` に置いてあります。詰まったときの答え合わせや、自分の手元と差分を見るのに使ってください。

```
hands-on/
├── README.md
├── step1/
│   └── main.py          # 最小のテキスト検索（30行程度）
├── step2/
│   └── main.py          # 同じ関数に画像を渡してクロスモーダル化
├── step3/
│   ├── main.py          # FastAPI 版（永続化あり）
│   └── static/
│       └── index.html   # 配布済みの Web UI(ハンズオン対象外)
├── step4/
│   └── main.py          # task_type / output_dimensionality 比較デモ
└── step5/
    ├── main.py          # step3 の DB を LanceDB に差し替えた版
    └── static/
        └── index.html   # step3 と同じ UI
```

---

## STEP 1: 最小のテキスト検索エンジン

完成品: `step1/main.py` ／ 実行: `uv run python hands-on/step1/main.py`

### 作るもの

数件のテキストをハードコードで登録し、クエリ文字列で類似テキストを検索する **30行程度のスクリプト** 1ファイル。

### コアになる3要素

```python
# (1) embed: テキストをベクトルに変換
def get_embedding(content):
    res = client.models.embed_content(model="gemini-embedding-2-preview", contents=content)
    return res.embeddings[0].values

# (2) similarity: 2つのベクトルの近さを測る
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# (3) search: クエリベクトルと全件のスコアを計算して上位を返す
scores = [(cosine_similarity(qv, item["vector"]), item) for item in db]
scores.sort(reverse=True)
top_k = scores[:3]
```

### 進め方

1. テキスト5〜10件を `db: list[dict]` にハードコードし、起動時に全件 embedding を取得して `vector` キーに保存する
2. クエリ文字列を1つ embedding に変換
3. 全件と cosine 類似度を計算 → 降順ソート → 上位3件を表示

### このSTEPで体感したいこと

- 「ベクトルDB」という言葉から想像するより、検索の本体は**驚くほど単純**（ベクトル化＋内積のソート）
- インデックス、永続化、フレームワークなどは**まだ何もいらない**

---

## STEP 2: クロスモーダル化

完成品: `step2/main.py` ／ 実行: `uv run python hands-on/step2/main.py`

### 作るもの

STEP1のスクリプトに画像を数枚追加。**同じ `get_embedding()` に PIL Image を渡すだけ** で画像もベクトル化できることを体験する。

### 差分はほぼこれだけ

```python
from PIL import Image

# テキストと同じ関数に画像を渡せる
img = Image.open("sunset.png")
vector = get_embedding(img)  # ← STEP1の関数そのまま
db.append({"type": "image", "content": "sunset.png", "vector": vector})
```

### 試してみたい5パターン

| クエリ | 検索対象 | 期待される挙動 |
|---|---|---|
| テキスト | テキスト | STEP1と同じ（ベースライン） |
| テキスト | 画像 | "sunset" で夕焼け画像がヒット |
| 画像 | 画像 | 似た色合い・構図の画像がヒット |
| 画像 | テキスト | 海の画像で "tropical beach" がヒット |
| **テキスト + 画像（マルチモーダル）** | 両方 | "日本" + ピンクのグラデーション画像 で 桜・富士山などがヒット |

### マルチモーダルクエリの組み立て方

クロスモーダル（どちらか一方）と区別して、**テキストと画像を1つのクエリ**として渡すには `types.Content` に複数の `Part` を入れます。

```python
from google.genai import types

content = types.Content(parts=[
    types.Part.from_text(text="日本"),
    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
])
res = client.models.embed_content(model=MODEL, contents=content)
vector = res.embeddings[0].values  # ← テキスト+画像を反映した単一ベクトル
```

`list[Part]` をそのまま渡すとバッチ扱い（各 Part ごとに別々の埋め込み）になります。1つの埋め込みにまとめたいときは **必ず `Content` でラップ** するのがポイント。

### このSTEPで体感したいこと

- Gemini Embedding 2 では、**テキストも画像も同じ3072次元の空間に乗る**
  - 実際に利用する場合は、latency と検索精度のトレードオフを意識しつつ次元削減を行うことが望ましい
- だから「コードはほぼ同じ」のまま、検索の表現力が一気に広がる
- ここが本ハンズオンの最大のポイント

### 補足: テスト用画像

簡単なグラデーション画像を Pillow で生成して使うのが手軽です（プロジェクトルートの `seed.py` に生成関数があるので参考にできます）。

---

## STEP 3: Web API + UI 化

完成品: `step3/main.py` ／ 実行: `uv run uvicorn hands-on.step3.main:app --reload` → http://localhost:8000

### 作るもの

STEP2 までを FastAPI でラップし、ブラウザから登録・検索ができる Web アプリにする。フロントエンド (`step3/static/index.html`) は配布済み なので、自分で書く必要はありません。ハンズオンでは「STEP2 の検索ロジックをどう HTTP エンドポイントに載せるか」だけに集中してください。

### 追加する要素

- **エンドポイント**
  - `POST /api/index/text` — テキスト登録
  - `POST /api/index/image` — 画像アップロード＆登録
  - `POST /api/search` — クエリ（テキスト or 画像）
- **永続化** — プロセス再起動でDBが消えないように、`pickle` で `db.pkl` に保存
- **静的UI** — `static/index.html` で登録フォームと検索フォームを置く

### 構造のポイント

```
hands-on/step3/
├── main.py          # STEP2 のロジックを FastAPI に載せ替える(ここを書く)
├── static/
│   └── index.html   # 登録・検索フォーム(配布済み)
├── uploads/         # アップロードした画像の保存先(実行時に作成)
└── db.pkl           # 永続化されたベクトルDB(実行時に作成)
```

### このSTEPで体感したいこと

- 検索ロジックそのものは STEP2 から変わっていない。変わるのは入出力の境界（HTTP・ファイル・ブラウザ）だけ
- "検索エンジン" を実用形にするときに何を足す必要があるか（永続化・アップロード処理・UI）が見える

---

## STEP 4: 実用化に向けた発展（任意）

完成品: `step4/main.py` ／ 実行: `uv run python hands-on/step4/main.py`

ここからは「持ち帰り課題」的な発展トピックです。`step4/main.py` は task_type と output_dimensionality をベースライン・最適化版・次元削減版の3パターンで比較するデモになっています。時間があれば手を動かしてみてください。

### 4-1. `task_type` を指定する

`embed_content` には `task_type` パラメータがあり、用途に応じて埋め込みを最適化できます。検索システムでは典型的に：

- ドキュメント側: `RETRIEVAL_DOCUMENT`
- クエリ側: `RETRIEVAL_QUERY`

を使い分けます。STEP3 までは省略していますが、指定すると検索精度が上がるケースがあります。

### 4-2. `output_dimensionality` で次元削減

Gemini Embedding 2 はデフォルト 3072 次元ですが、より低い次元（例: 768, 1536）を指定できます。**メモリ・ストレージ・類似度計算のコストはすべて次元数に比例** するので、件数が増えるほど効きます。

### 4-3. 本物のベクトルDBへ置き換え

`list[dict]` での全件スキャンは件数が増えると線形に遅くなります。実用では：

- pgvector（PostgreSQL拡張）
- Qdrant / Milvus / Weaviate
- LanceDB（埋め込み型・SQLite 的）
- Vertex AI Vector Search

などの ANN（近似最近傍）インデックスに置き換えます。**インターフェース（embed → 近傍検索）は同じ** なので、STEP3 の `db: list[dict]` の部分を差し替えるだけで移行できる、というのが感覚としてつかめると良いです。具体例として **STEP 5** に LanceDB 版の完成品があります。

### 4-4. ハイブリッド検索

ベクトル検索はセマンティックには強いですが、固有名詞や型番のような完全一致は苦手です。BM25 のようなキーワード検索とスコアを組み合わせる **ハイブリッド検索** が実プロダクトでは一般的です。

---

## STEP 5: 本物のベクトル DB に置き換え（LanceDB）

完成品: `step5/main.py` ／ 実行: `uv run uvicorn hands-on.step5.main:app --reload` → http://localhost:8000

### 作るもの

STEP3 の **検索ロジックの外形は変えず**、内部の `_db: list[dict]` を **LanceDB** に置き換えます。LanceDB は Rust 製の埋め込み型（SQLite 的）ベクトル DB で、`pip install lancedb` だけで使えて、永続化と ANN 検索を肩代わりしてくれます。

### 差分のかたち

```python
import lancedb
import pyarrow as pa

_lance = lancedb.connect("./lance_db")
_table = _lance.create_table(
    "items",
    schema=pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 3072)),
        pa.field("type", pa.string()),
        pa.field("content", pa.string()),
        pa.field("image_url", pa.string()),
    ]),
    exist_ok=True,
)

# 登録
_table.add([{"vector": vec, "type": "text", "content": "...", "image_url": None}])

# 検索（コサイン類似度の上位N件）
hits = _table.search(query_vec).metric("cosine").limit(20).to_list()
# hit["_distance"] は 1 - cosine_similarity なので、表示用は score = 1 - _distance
```

### 構造のポイント

```
hands-on/step5/
├── main.py        # step3 から DB 周りだけ差し替え
├── static/
│   └── index.html # step3 と同じ UI(無改修)
├── uploads/       # アップロード画像(実行時に作成)
└── lance_db/      # LanceDB のデータ(実行時に作成)
```

### このSTEPで体感したいこと

- **エンドポイントの外形・UI・検索の意味論はSTEP3と完全に同じ**。差し替わるのは DB 周りだけ。
- 永続化（`db.pkl` の管理）が消えて、DB が肩代わりしてくれる。
- 件数が増えても、`_table.create_index(...)` 1行で ANN インデックスを足せる（小規模では sequential scan で十分）。

### つまづきところ

- スキーマで `vector` の **固定次元** を指定する必要がある。`output_dimensionality` を変えるときは schema も合わせる。
- `metric("cosine")` の `_distance` は **距離（小さいほど類似）**。UI 表示用の **類似度スコア** に直すには `1 - _distance`。
- `text_results` と `image_results` を上位5件ずつ返すために、`limit(20)` で多めに取ってから振り分けている（型ごとの top-k フィルタを DB 側に押し込む書き方も可能）。

---

## 参考: 必要な準備

```bash
# プロジェクトルートで一度だけ
uv sync
cp .env.example .env  # GEMINI_API_KEY を設定
```

各STEPの実行コマンドは STEP ごとの見出しに記載しています（プロジェクトルートから実行する想定）。

---
---

# Gemini Embedding Hands-on: Build a Cross-Modal Search System (English)

A hands-on tutorial for building a search system, aimed at anyone who has tried the basics of the Gemini Embedding API (`embed_content` returning a vector) at least once. If you haven't, there's a Colab for first-timers linked from the root `README.md` of this repo — start there.

The goal is to get a **"cross-modal search system that accepts both text and image queries"** running with your own hands, building up the concepts one at a time.

---

## Overview

| STEP | What you build | What you learn |
|---|---|---|
| 1 | Minimum text search engine (one file, ~30 lines) | Search = nearest-neighbor lookup in a vector space |
| 2 | Cross-modal (pass an image to the same function) | Text and image live in the same vector space |
| 3 | Web API + UI (FastAPI + persistence) | How to wire a search engine into a "usable" form |
| 4 | Steps toward production (optional) | Bridges to production usage |
| 5 | Swap in a real vector DB (LanceDB) | `list[dict]` → a dedicated DB is a drop-in replacement |

Each STEP adds the **minimum diff** on top of the previous one. STEPs 1–2 are single-file scripts; STEP 3 is where a web framework first appears.

## Directory layout

The completed reference for each STEP lives in `step1/` through `step5/`. Use them as the answer key when you get stuck or to diff against your own work.

```
hands-on/
├── README.md
├── step1/
│   └── main.py          # Minimum text search (~30 lines)
├── step2/
│   └── main.py          # Same function with an image: cross-modal
├── step3/
│   ├── main.py          # FastAPI version (with persistence)
│   └── static/
│       └── index.html   # Pre-built web UI (out of scope for the hands-on)
├── step4/
│   └── main.py          # task_type / output_dimensionality comparison
└── step5/
    ├── main.py          # step3 with the DB swapped to LanceDB
    └── static/
        └── index.html   # Same UI as step3
```

---

## STEP 1: Minimum text search engine

Reference: `step1/main.py` ／ Run: `uv run python hands-on/step1/main.py`

### What to build

A **~30-line script** in a single file that registers a handful of hard-coded texts and finds similar ones for a query string.

### The three core pieces

```python
# (1) embed: turn text into a vector
def get_embedding(content):
    res = client.models.embed_content(model="gemini-embedding-2-preview", contents=content)
    return res.embeddings[0].values

# (2) similarity: measure how close two vectors are
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# (3) search: score every item against the query and return the top-k
scores = [(cosine_similarity(qv, item["vector"]), item) for item in db]
scores.sort(reverse=True)
top_k = scores[:3]
```

### How to proceed

1. Hard-code 5–10 strings in `db: list[dict]`, embed them all at startup, and store the result under a `vector` key.
2. Embed a single query string.
3. Compute cosine similarity against every item → sort descending → print the top 3.

### What to take away from this STEP

- The core of search is **surprisingly simple** (vectorize + sort by dot product) — much less than what "vector DB" makes it sound.
- No index, no persistence, no framework — **none of that is needed yet**.

---

## STEP 2: Cross-modal

Reference: `step2/main.py` ／ Run: `uv run python hands-on/step2/main.py`

### What to build

Add a few images on top of STEP 1's script. Experience how **just passing a `PIL.Image` to the same `get_embedding()` function** is all it takes to embed an image too.

### The whole diff looks like this

```python
from PIL import Image

# Pass an image to the same function used for text
img = Image.open("sunset.png")
vector = get_embedding(img)  # ← exact same function as STEP 1
db.append({"type": "image", "content": "sunset.png", "vector": vector})
```

### Five query patterns to try

| Query | Searched against | Expected behavior |
|---|---|---|
| Text | Text | Same as STEP 1 (baseline) |
| Text | Image | "sunset" hits the sunset image |
| Image | Image | Visually similar images rise to the top |
| Image | Text | An ocean image hits "tropical beach" |
| **Text + Image (multimodal)** | Both | "日本" + a pink gradient image hits 桜 / 富士山 |

### How to assemble a multimodal query

To pass **text and image as a single query** (as opposed to cross-modal, where it's one or the other), wrap multiple `Part`s in a `types.Content`:

```python
from google.genai import types

content = types.Content(parts=[
    types.Part.from_text(text="日本"),
    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
])
res = client.models.embed_content(model=MODEL, contents=content)
vector = res.embeddings[0].values  # ← a single vector reflecting text + image
```

Passing a bare `list[Part]` is treated as a **batch** (one separate embedding per Part). To collapse them into a single embedding, **wrap them in a `Content`** — that's the key.

### What to take away from this STEP

- With Gemini Embedding 2, **text and images live in the same 3072-dimensional space**.
  - In practice you'll want to balance latency vs. retrieval quality and reduce the dimensionality.
- Because of that, "the code stays almost the same" while the expressive range of search jumps dramatically.
- This is the central insight of the whole tutorial.

### Side note: test images

Generating gradient images with Pillow is the easiest way to get started (there's a helper in the project root's `seed.py` you can reference).

---

## STEP 3: Web API + UI

Reference: `step3/main.py` ／ Run: `uv run uvicorn hands-on.step3.main:app --reload` → http://localhost:8000

### What to build

Wrap STEP 2's logic in FastAPI so the registration/search flows are reachable from a browser. The frontend (`step3/static/index.html`) **ships pre-built**, so you don't need to write it. Focus the hands-on time purely on "how do I expose STEP 2's search logic as HTTP endpoints?"

### What's added

- **Endpoints**
  - `POST /api/index/text` — register text
  - `POST /api/index/image` — upload and register an image
  - `POST /api/search` — query (text or image)
- **Persistence** — save to `db.pkl` via `pickle` so the DB survives process restarts.
- **Static UI** — `static/index.html` holds the registration and search forms.

### Structural notes

```
hands-on/step3/
├── main.py          # STEP 2's logic ported to FastAPI (this is what you write)
├── static/
│   └── index.html   # Registration & search forms (pre-built)
├── uploads/         # Uploaded image storage (created at runtime)
└── db.pkl           # Persistent vector DB (created at runtime)
```

### What to take away from this STEP

- The search logic itself doesn't change from STEP 2. What changes is the I/O boundary (HTTP, files, browser).
- You can see what it takes to turn a "search engine" into something usable: persistence, upload handling, UI.

---

## STEP 4: Toward production (optional)

Reference: `step4/main.py` ／ Run: `uv run python hands-on/step4/main.py`

This part is a "take-home" set of follow-ups. `step4/main.py` compares three configurations — baseline, optimized, and dimensionality-reduced — for `task_type` and `output_dimensionality`. Try it if you have time.

### 4-1. Specify `task_type`

`embed_content` takes a `task_type` parameter that optimizes the embedding for a given use case. For search systems, you typically use:

- Document side: `RETRIEVAL_DOCUMENT`
- Query side: `RETRIEVAL_QUERY`

STEPs 1–3 omit this, but specifying it can improve retrieval quality.

### 4-2. Reduce dimensions with `output_dimensionality`

Gemini Embedding 2 defaults to 3072 dimensions, but you can request smaller (e.g. 768 or 1536). **Memory, storage, and similarity-computation costs all scale linearly with the dimension**, so the savings grow with corpus size.

### 4-3. Swap in a real vector DB

Scanning a `list[dict]` end-to-end gets linearly slower as the corpus grows. In production you'd swap in something like:

- pgvector (PostgreSQL extension)
- Qdrant / Milvus / Weaviate
- LanceDB (embedded, SQLite-style)
- Vertex AI Vector Search

…or another ANN (approximate nearest neighbor) index. **The interface (embed → nearest-neighbor → top-k) stays the same**, so migrating is essentially replacing STEP 3's `db: list[dict]` block. As a concrete example, **STEP 5** is exactly that — a LanceDB-backed copy.

### 4-4. Hybrid search

Vector search is strong on semantic similarity but weak on exact matches like proper nouns or model numbers. Production systems often pair it with a keyword search (e.g. BM25) and merge the scores — **hybrid search**.

---

## STEP 5: Swap in a real vector DB (LanceDB)

Reference: `step5/main.py` ／ Run: `uv run uvicorn hands-on.step5.main:app --reload` → http://localhost:8000

### What to build

Keep STEP 3's **search logic untouched** but swap the internal `_db: list[dict]` for **LanceDB**. LanceDB is a Rust-based embedded vector DB (SQLite-style) — `pip install lancedb` is all you need, and it handles persistence and ANN search for you.

### Shape of the diff

```python
import lancedb
import pyarrow as pa

_lance = lancedb.connect("./lance_db")
_table = _lance.create_table(
    "items",
    schema=pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 3072)),
        pa.field("type", pa.string()),
        pa.field("content", pa.string()),
        pa.field("image_url", pa.string()),
    ]),
    exist_ok=True,
)

# Insert
_table.add([{"vector": vec, "type": "text", "content": "...", "image_url": None}])

# Search (top-N by cosine similarity)
hits = _table.search(query_vec).metric("cosine").limit(20).to_list()
# hit["_distance"] is 1 - cosine_similarity, so for display: score = 1 - _distance
```

### Structural notes

```
hands-on/step5/
├── main.py        # step3 with only the DB layer swapped out
├── static/
│   └── index.html # Same UI as step3 (unchanged)
├── uploads/       # Uploaded images (created at runtime)
└── lance_db/      # LanceDB data (created at runtime)
```

### What to take away from this STEP

- **The endpoint shapes, the UI, and the semantics of search are identical to STEP 3.** Only the DB layer is swapped.
- The persistence chore (managing `db.pkl`) disappears — the DB handles it for you.
- As the corpus grows, you can add an ANN index with a single `_table.create_index(...)` call (sequential scan is enough at small scale).

### Gotchas

- The schema requires the `vector` field to have a **fixed dimension**. If you change `output_dimensionality`, change the schema too.
- With `metric("cosine")`, `_distance` is a **distance (smaller = closer)**. Convert it to a **similarity score** for display with `1 - _distance`.
- To return up to 5 of each type (`text_results` / `image_results`), the script fetches a wider `limit(20)` and partitions afterward. You could also push per-type top-k filtering down to the DB.

---

## Appendix: setup you need

```bash
# Run once from the project root
uv sync
cp .env.example .env  # set GEMINI_API_KEY
```

The run command for each STEP is listed under that STEP's heading (run from the project root).
