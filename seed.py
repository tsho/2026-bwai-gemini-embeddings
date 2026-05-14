"""デモデータ投入スクリプト.

サーバー (``main.py``) が起動している前提で、テキスト約30件と画像約20件を
API 経由で登録する。画像は Pillow で生成したグラデーション画像を使う。

Example:
    $ uv run uvicorn main:app --reload  # 別ターミナルで先に起動
    $ uv run python seed.py
"""

from __future__ import annotations

import argparse
import io
import sys
import time

import httpx
from PIL import Image, ImageDraw, ImageFont

_BASE_URL = "http://localhost:8000"
# Vertex AI の gemini-embedding-2 クォータが 5 RPM (= 12 秒/件) のため、
# 1件ごとに少しマージンを取って 13 秒スリープする。
_RATE_LIMIT_SECONDS = 13
# 念のため、quota 起因の失敗 (429 / 500 / 503) は数回までリトライする。
_MAX_RETRIES = 3
_RETRY_WAIT_SECONDS = 60
# デフォルトはハンズオン向けに少数件。`--full` で全件投入。
_DEFAULT_TEXTS = 6
_DEFAULT_IMAGES = 4

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


def _post_with_retry(client: httpx.Client, path: str, **kwargs) -> httpx.Response:
    """API を叩き、quota 起因の失敗 (429/500/503) は待ってリトライする.

    Args:
        client: 接続済みの httpx クライアント。
        path: ``/api/index/text`` などのエンドポイントパス。
        **kwargs: ``client.post`` にそのまま渡す引数 (``data`` / ``files`` など)。

    Returns:
        正常応答 (2xx) の ``httpx.Response``。

    Raises:
        httpx.HTTPStatusError: リトライ上限を超えても 2xx にならなかった場合。
    """
    for attempt in range(_MAX_RETRIES):
        res = client.post(path, **kwargs)
        if res.status_code in (429, 500, 503) and attempt < _MAX_RETRIES - 1:
            print(
                f"    got {res.status_code}, sleeping {_RETRY_WAIT_SECONDS}s before retry "
                f"({attempt + 1}/{_MAX_RETRIES - 1})..."
            )
            time.sleep(_RETRY_WAIT_SECONDS)
            continue
        res.raise_for_status()
        return res
    raise RuntimeError("unreachable")  # _MAX_RETRIES > 0 なら必ず return か raise


def main(*, full: bool = False) -> None:
    """``_TEXTS`` と ``_IMAGES`` を順番に API へ投入する.

    Vertex AI の embedding クォータ (5 RPM) に収まるよう、1件ごとに
    ``_RATE_LIMIT_SECONDS`` 秒のスリープを挟む。失敗時は
    ``_post_with_retry`` がリトライする。

    Args:
        full: ``True`` なら ``_TEXTS`` / ``_IMAGES`` を丸ごと使う (合計 50 件)。
            既定は ``False`` で、ハンズオン向けに ``_DEFAULT_TEXTS`` /
            ``_DEFAULT_IMAGES`` 件に絞る。

    サーバーが起動していなければ案内メッセージを出して終了する。
    """
    texts = _TEXTS if full else _TEXTS[:_DEFAULT_TEXTS]
    images = _IMAGES if full else _IMAGES[:_DEFAULT_IMAGES]

    client = httpx.Client(base_url=_BASE_URL, timeout=60)

    # Check server is running
    try:
        client.get("/")
    except httpx.ConnectError:
        print(f"Error: Server is not running at {_BASE_URL}")
        print("Start it first: uv run uvicorn main:app --reload")
        sys.exit(1)

    total = len(texts) + len(images)
    count = 0

    eta_minutes = (total * _RATE_LIMIT_SECONDS) / 60
    mode = "full" if full else "default"
    print(
        f"Registering {total} demo items ({mode}) at ~{60 // _RATE_LIMIT_SECONDS} RPM "
        f"(sleeping {_RATE_LIMIT_SECONDS}s between each, ETA ~{eta_minutes:.1f} min)..."
    )

    for text in texts:
        count += 1
        print(f"  [{count}/{total}] text: {text[:40]}...")
        _post_with_retry(client, "/api/index/text", data={"text": text})
        if count < total:
            time.sleep(_RATE_LIMIT_SECONDS)

    for name, c1, c2, label in images:
        count += 1
        print(f"  [{count}/{total}] image: {label}")
        img_bytes = generate_image(c1, c2, label)
        _post_with_retry(
            client,
            "/api/index/image",
            files={"file": (f"{name}.png", img_bytes, "image/png")},
        )
        if count < total:
            time.sleep(_RATE_LIMIT_SECONDS)

    print(f"Done! {total} items registered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Seed demo items into the running cross-modal search server. "
            "Defaults to a small subset suitable for the 5 RPM quota; "
            "pass --full to register all 50 items (takes ~11 minutes)."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full dataset (all _TEXTS / _IMAGES, ~50 items)",
    )
    args = parser.parse_args()
    main(full=args.full)
