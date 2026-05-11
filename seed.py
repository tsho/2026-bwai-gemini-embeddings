"""デモデータ投入スクリプト.

サーバー (``main.py``) が起動している前提で、テキスト約30件と画像約20件を
API 経由で登録する。画像は Pillow で生成したグラデーション画像を使う。

Example:
    $ uv run uvicorn main:app --reload  # 別ターミナルで先に起動
    $ uv run python seed.py
"""

from __future__ import annotations

import io
import sys

import httpx
from PIL import Image, ImageDraw, ImageFont

_BASE_URL = "http://localhost:8000"

_TEXTS = [
    "東京タワーは1958年に完成した電波塔で、高さは333メートルです",
    "富士山は日本最高峰の山で、標高3776メートルです",
    "桜は日本の春を象徴する花で、3月から4月にかけて咲きます",
    "新幹線は時速300キロ以上で走る日本の高速鉄道です",
    "寿司は酢飯の上に新鮮な魚介類をのせた日本の伝統料理です",
    "京都には数多くの寺社仏閣があり、世界遺産にも登録されています",
    "相撲は日本の国技とされる格闘技で、土俵の上で行われます",
    "北海道はラベンダー畑や雪まつりで有名な日本最北の島です",
    "抹茶は茶道で使われる粉末状の緑茶で、独特の苦味があります",
    "浮世絵は江戸時代に発展した日本の木版画芸術です",
    "A golden retriever playing fetch in a sunny park",
    "A cat sleeping on a warm windowsill in the afternoon",
    "A tropical beach with turquoise water and white sand",
    "A snowy mountain peak with clear blue sky",
    "A bustling city street at night with neon lights",
    "A plate of fresh pasta with tomato sauce and basil",
    "A cup of hot coffee with latte art on a wooden table",
    "A field of sunflowers stretching to the horizon",
    "A red sports car driving on a winding mountain road",
    "An astronaut floating in space with Earth in the background",
    "A cozy library filled with old leather-bound books",
    "A rainbow appearing after a rainstorm over green hills",
    "A violin and sheet music on a grand piano",
    "A campfire under a starry night sky in the forest",
    "A colorful coral reef with tropical fish underwater",
    "プログラミングはコンピュータに命令を与えるための技術です",
    "機械学習はデータからパターンを学習するAIの一分野です",
    "地球温暖化は二酸化炭素の増加により平均気温が上昇する現象です",
    "ピカソはキュビスムを代表するスペイン出身の画家です",
    "モーツァルトは古典派音楽を代表するオーストリアの作曲家です",
]

_IMAGES = [
    ("sunset", "#FF6B35", "#FFC300", "Sunset"),
    ("ocean", "#006994", "#40E0D0", "Ocean"),
    ("forest", "#228B22", "#90EE90", "Forest"),
    ("fire", "#FF4500", "#FFD700", "Fire"),
    ("night_sky", "#0C1445", "#4169E1", "Night Sky"),
    ("cherry_blossom", "#FFB7C5", "#FF69B4", "Cherry Blossom"),
    ("snow", "#E8E8E8", "#FFFFFF", "Snow"),
    ("desert", "#EDC9AF", "#DAA520", "Desert"),
    ("lavender", "#9370DB", "#E6E6FA", "Lavender Field"),
    ("autumn", "#FF8C00", "#8B4513", "Autumn Leaves"),
    ("rose", "#FF007F", "#C71585", "Red Rose"),
    ("sky", "#87CEEB", "#F0F8FF", "Blue Sky"),
    ("earth", "#4169E1", "#228B22", "Earth"),
    ("volcano", "#8B0000", "#FF4500", "Volcano"),
    ("coral", "#FF7F50", "#00CED1", "Coral Reef"),
    ("mountain", "#708090", "#B0C4DE", "Mountain"),
    ("meadow", "#7CFC00", "#FFFF00", "Meadow"),
    ("storm", "#2F4F4F", "#778899", "Storm"),
    ("rainbow", "#FF0000", "#9400D3", "Rainbow"),
    ("city", "#36454F", "#FFD700", "City Lights"),
]


def generate_image(color1: str, color2: str, label: str) -> bytes:
    """ラベル文字を中央に重ねた縦方向グラデーション画像を生成する.

    Args:
        color1: 上端の色 (CSSカラー文字列、例: ``"#FF6B35"``)。
        color2: 下端の色 (CSSカラー文字列)。
        label: 画像中央に描画する英字ラベル。

    Returns:
        PNG エンコード済みのバイト列。
    """
    w, h = 256, 256
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    r1, g1, b1 = Image.new("RGB", (1, 1), color1).getpixel((0, 0))
    r2, g2, b2 = Image.new("RGB", (1, 1), color2).getpixel((0, 0))

    for y in range(h):
        t = y / h
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, (h - th) / 2), label, fill="white", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    """``_TEXTS`` と ``_IMAGES`` を順番に API へ投入する.

    サーバーが起動していなければ案内メッセージを出して終了する。
    """
    client = httpx.Client(base_url=_BASE_URL, timeout=60)

    # Check server is running
    try:
        client.get("/")
    except httpx.ConnectError:
        print(f"Error: Server is not running at {_BASE_URL}")
        print("Start it first: uv run uvicorn main:app --reload")
        sys.exit(1)

    total = len(_TEXTS) + len(_IMAGES)
    count = 0

    print(f"Registering {total} demo items...")

    for text in _TEXTS:
        count += 1
        print(f"  [{count}/{total}] text: {text[:40]}...")
        res = client.post("/api/index/text", data={"text": text})
        res.raise_for_status()

    for name, c1, c2, label in _IMAGES:
        count += 1
        print(f"  [{count}/{total}] image: {label}")
        img_bytes = generate_image(c1, c2, label)
        res = client.post(
            "/api/index/image",
            files={"file": (f"{name}.png", img_bytes, "image/png")},
        )
        res.raise_for_status()

    print(f"Done! {total} items registered.")


if __name__ == "__main__":
    main()
