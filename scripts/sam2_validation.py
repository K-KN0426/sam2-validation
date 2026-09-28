"""SAM2 (Segment Anything Model 2) の動作検証スクリプト。

指定した画像とプロンプト点（クリック座標）に対してセグメンテーションマスクを
生成し、オーバーレイ画像として保存する。モデルは Hugging Face Hub 上の
`facebook/sam2-hiera-tiny` を使用する。

使い方:
    python scripts/sam2_validation.py \
        --image assets/sample.png \
        --point 320 240 \
        --output outputs/sample_mask.png

必要パッケージ:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="入力画像パス")
    parser.add_argument(
        "--point",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        required=True,
        help="前景プロンプトとして与えるピクセル座標",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="マスクオーバーレイの保存先"
    )
    parser.add_argument(
        "--model-id",
        default="facebook/sam2-hiera-tiny",
        help="Hugging Face Hub 上のモデルID",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import numpy as np
    from PIL import Image

    image = Image.open(args.image).convert("RGB")

    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except ImportError as exc:
        print(f"[NG] 依存パッケージの import に失敗しました: {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] モデルをロードします: {args.model_id}")
    load_start = time.monotonic()
    try:
        processor = Sam2Processor.from_pretrained(args.model_id)
        model = Sam2Model.from_pretrained(args.model_id)
    except Exception as exc:  # noqa: BLE001 - 検証目的で詳細を出力する
        print(f"[NG] モデルのロードに失敗しました: {exc}", file=sys.stderr)
        return 1
    print(f"[INFO] モデルロード完了 ({time.monotonic() - load_start:.1f}s)")

    inputs = processor(
        images=image,
        input_points=[[[list(args.point)]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )

    infer_start = time.monotonic()
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"[INFO] 推論完了 ({time.monotonic() - infer_start:.1f}s)")

    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
    )
    best_mask_idx = int(outputs.iou_scores.argmax())
    mask = masks[0][0][best_mask_idx].numpy().astype(bool)

    overlay = np.array(image).copy()
    overlay[mask] = (overlay[mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(
        np.uint8
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    print(f"[OK] マスクオーバーレイを保存しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
