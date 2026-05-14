# Gemini Embedding ハンズオン: クロスモーダル検索システムを作る

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

### 罠メモ

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
