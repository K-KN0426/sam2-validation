"""検証用のサンプル画像を生成するスクリプト（外部ダウンロード不要）。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "assets" / "sample.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (640, 480), color=(235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.ellipse((200, 140, 440, 340), fill=(30, 120, 200))
    draw.rectangle((60, 320, 220, 440), fill=(200, 60, 60))
    draw.polygon([(480, 380), (560, 260), (620, 400)], fill=(60, 160, 60))

    image.save(out_path)
    print(f"[OK] サンプル画像を作成しました: {out_path}")


if __name__ == "__main__":
    main()
