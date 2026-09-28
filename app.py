"""SAM2 のクリックセグメンテーションを試す Gradio UI。

画像をアップロードしてクリックすると、その位置を点プロンプトとして SAM2 で推論し、
マスクを重ねて表示する。

- まとめて選択してから実行: クリック位置を貯め、「選択を実行」でまとめて推論する。
  OFF に戻すと、貯まっているクリック位置を推論する。
- 外枠の線で表示: マスクを輪郭線で描画する。推論はやり直さない。
- クリックした塊だけ残す（既定 ON）: オープニング処理後、クリック点を含む連結成分
  だけを残して穴を埋める。元のマスクを保持しているため、切り替えで推論はやり直さない。

右側には、マスク一覧（Gradio AnnotatedImage）、クリック座標、輪郭のポリゴン座標を
表示する。輪郭は OpenCV の findContours でマスクから計算する。

使い方:
    pip install -r requirements.txt
    python app.py

ブラウザで http://127.0.0.1:7860 を開く。
"""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

MODEL_ID = "facebook/sam2-hiera-tiny"

# クリックした順に使う塗りつぶし色
COLORS = [
    (255, 0, 0),
    (0, 160, 255),
    (0, 200, 0),
    (255, 200, 0),
    (200, 0, 255),
    (0, 220, 220),
]

# 輪郭を単純化する際の許容誤差（輪郭の全周に対する割合）
CONTOUR_APPROX_RATIO = 0.005

# 「外枠の線で表示」モードで描く線の太さ（px）
OUTLINE_THICKNESS = 3

# 「クリックした塊だけ残す」で使うオープニングのカーネルサイズ（px）。
# 大きくするほど小さな飛び地が消えるが、細い部分も削れやすくなる。
CLEAN_KERNEL_SIZE = 7

print(f"[INFO] モデルをロードします: {MODEL_ID}（初回はダウンロードが発生します）")
processor = Sam2Processor.from_pretrained(MODEL_ID)
model = Sam2Model.from_pretrained(MODEL_ID)
model.eval()
print("[INFO] モデルのロードが完了しました")


def predict_best_mask(image: Image.Image, x: int, y: int) -> np.ndarray:
    inputs = processor(
        images=image,
        input_points=[[[[x, y]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])
    candidate_masks = masks[0][0]  # (num_masks, H, W)
    scores = outputs.iou_scores[0][0]  # (num_masks,)
    best_idx = int(scores.argmax())
    return candidate_masks[best_idx].numpy().astype(bool)


def clean_mask(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """SAM2 のマスクから、クリック点を含まない小さな飛び地を取り除く。

    1. オープニングで細い線や小さな点を削り、飛び地とのつながりを切る
    2. クリック点を含む連結成分だけを残す（含まなければ最大の成分）
    3. 残った塊の中の穴を埋める
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLEAN_KERNEL_SIZE,) * 2)
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num <= 1:
        return mask  # 削りすぎて何も残らない場合は元のマスクを返す
    keep = labels[y, x]
    if keep == 0:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == keep).astype(np.uint8)

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def finalize_point(point: dict, clean: bool) -> dict:
    """生のマスクから、表示用のマスクと外枠座標を作る。"""
    raw = point["raw_mask"]
    mask = clean_mask(raw, point["x"], point["y"]) if clean else raw
    return {**point, "mask": mask, "outline": mask_to_outline(mask)}


def mask_to_outline(mask: np.ndarray) -> list[list[int]]:
    """マスクの外枠（最も面積の大きい輪郭）を座標のリストとして返す。"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = CONTOUR_APPROX_RATIO * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    return simplified.reshape(-1, 2).tolist()


def mask_contours(mask: np.ndarray):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return contours


def mask_to_outline_band(mask: np.ndarray) -> np.ndarray:
    """マスクの外枠だけを線として描いた bool 配列を返す（AnnotatedImage 用）。"""
    band = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(band, mask_contours(mask), -1, 1, thickness=OUTLINE_THICKNESS)
    return band.astype(bool)


def render(
    original: Image.Image,
    points: list[dict],
    pending: list[tuple[int, int]],
    outline_mode: bool = False,
) -> Image.Image:
    arr = np.array(original.convert("RGB")).copy()
    for point in points:
        color = COLORS[point["index"] % len(COLORS)]
        mask = point["mask"]
        if outline_mode:
            cv2.drawContours(arr, mask_contours(mask), -1, color, thickness=OUTLINE_THICKNESS)
        else:
            arr[mask] = (arr[mask] * 0.4 + np.array(color) * 0.6).astype(np.uint8)

    for offset, (x, y) in enumerate(pending):
        label = str(len(points) + offset + 1)
        cv2.circle(arr, (x, y), 8, (255, 255, 255), thickness=-1)
        cv2.circle(arr, (x, y), 8, (0, 0, 0), thickness=2)
        cv2.putText(
            arr, label, (x + 10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )

    return Image.fromarray(arr)


def points_to_rows(points: list[dict]) -> list[list[int]]:
    return [
        [p["index"] + 1, p["x"], p["y"], len(p["outline"])] for p in points
    ]


def points_to_outlines(points: list[dict]) -> dict:
    return {f"#{p['index'] + 1} ({p['x']}, {p['y']})": p["outline"] for p in points}


def points_to_annotations(original: Image.Image, points: list[dict], outline_mode: bool = False):
    if original is None:
        return None
    annotations = [
        (
            mask_to_outline_band(p["mask"]) if outline_mode else p["mask"],
            f"#{p['index'] + 1} ({p['x']}, {p['y']})",
        )
        for p in points
    ]
    return (original, annotations)


def on_upload(image: Image.Image):
    if image is None:
        return None, None, [], [], [], None, {}
    return image, image, [], [], [], (image, []), {}


def run_pending(
    original: Image.Image,
    points: list[dict],
    pending: list[tuple[int, int]],
    clean: bool,
):
    """貯まった候補点をまとめてSAM2に投げ、マスクとして確定させる。"""
    for x, y in pending:
        raw_mask = predict_best_mask(original, x, y)
        point = {"index": len(points), "x": x, "y": y, "raw_mask": raw_mask}
        points = points + [finalize_point(point, clean)]
    return points


def outputs_for(
    original: Image.Image,
    points: list[dict],
    pending: list[tuple[int, int]],
    outline_mode: bool = False,
):
    overlay = render(original, points, pending, outline_mode) if original is not None else None
    return (
        overlay,
        points,
        pending,
        points_to_rows(points),
        points_to_annotations(original, points, outline_mode),
        points_to_outlines(points),
    )


def on_select(
    original: Image.Image,
    points: list[dict],
    pending: list[tuple[int, int]],
    batch_mode: bool,
    outline_mode: bool,
    clean: bool,
    evt: gr.SelectData,
):
    if original is None:
        return gr.skip(), gr.skip(), pending, gr.skip(), gr.skip(), gr.skip()

    x, y = evt.index
    if batch_mode:
        # 候補点を打つだけ。マスク一覧・表・JSONは変化しないので再送しない
        # （毎回まとめて送ると重いマスク画像まで再転送されて重くなるため）。
        pending = pending + [(x, y)]
        overlay = render(original, points, pending, outline_mode)
        return overlay, gr.skip(), pending, gr.skip(), gr.skip(), gr.skip()

    points = run_pending(original, points, [(x, y)], clean)
    return outputs_for(original, points, [], outline_mode)


def on_run(
    original: Image.Image,
    points: list[dict],
    pending: list[tuple[int, int]],
    outline_mode: bool,
    clean: bool,
):
    if original is None or not pending:
        return outputs_for(original, points, [], outline_mode)
    points = run_pending(original, points, pending, clean)
    return outputs_for(original, points, [], outline_mode)


def on_toggle_batch(
    batch_mode: bool,
    original: Image.Image,
    points: list[dict],
    pending: list[tuple[int, int]],
    outline_mode: bool,
    clean: bool,
):
    if batch_mode or original is None or not pending:
        # バッチモードON、または貯まった候補点が無ければ何もしない
        return (gr.update(visible=batch_mode),) + outputs_for(
            original, points, pending, outline_mode
        )

    # バッチモードOFFに戻した時点で、貯まっていた候補点を自動実行する
    points = run_pending(original, points, pending, clean)
    return (gr.update(visible=batch_mode),) + outputs_for(original, points, [], outline_mode)


def on_toggle_clean(
    clean: bool,
    original: Image.Image,
    points: list[dict],
    pending: list[tuple[int, int]],
    outline_mode: bool,
):
    """「クリックした塊だけ残す」のON/OFF。生マスクを保持しているので推論はやり直さない。"""
    points = [finalize_point(p, clean) for p in points]
    return outputs_for(original, points, pending, outline_mode)


def on_toggle_outline(
    outline_mode: bool,
    original: Image.Image,
    points: list[dict],
    pending: list[tuple[int, int]],
):
    """塗りつぶし / 外枠の線 を切り替える。推論はやり直さず描画だけ更新する。"""
    if original is None:
        return gr.skip(), gr.skip()
    return (
        render(original, points, pending, outline_mode),
        points_to_annotations(original, points, outline_mode),
    )


def on_reset(original: Image.Image, outline_mode: bool):
    return outputs_for(original, [], [], outline_mode)


with gr.Blocks(title="SAM2 検証UI") as demo:
    gr.Markdown(
        "# SAM2 クリックセグメンテーション検証\n"
        "画像をアップロードしてクリックすると、その物体が塗りつぶされます。"
        f"複数回クリックすると別の色で追加されます。モデル: `{MODEL_ID}`\n\n"
        "右側の一覧にマウスを乗せると、対応するマスクがハイライトされます。"
    )
    with gr.Row():
        batch_toggle = gr.Checkbox(
            label="まとめて選択してから実行", value=False
        )
        outline_toggle = gr.Checkbox(
            label="外枠の線で表示（OFFで塗りつぶし）", value=False
        )
        clean_toggle = gr.Checkbox(
            label="クリックした塊だけ残す", value=True
        )
    original_state = gr.State(None)
    points_state = gr.State([])
    pending_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            image_display = gr.Image(type="pil", label="画像（クリックで物体を塗りつぶす）")
            with gr.Row():
                run_btn = gr.Button("選択を実行", variant="primary", visible=False)
                reset_btn = gr.Button("選択をリセット")
        with gr.Column(scale=2):
            mask_gallery = gr.AnnotatedImage(
                label="選択したマスク（ホバーでハイライト）",
                show_legend=True,
            )
            points_table = gr.Dataframe(
                headers=["#", "X", "Y", "輪郭の点数"],
                datatype=["number", "number", "number", "number"],
                label="選択した座標",
            )
            outline_json = gr.JSON(label="マスクの外枠座標（輪郭のポリゴン点）")

    result_outputs = [
        image_display,
        points_state,
        pending_state,
        points_table,
        mask_gallery,
        outline_json,
    ]

    image_display.upload(
        on_upload,
        inputs=image_display,
        outputs=[original_state] + result_outputs,
    )
    image_display.select(
        on_select,
        inputs=[
            original_state, points_state, pending_state, batch_toggle, outline_toggle, clean_toggle,
        ],
        outputs=result_outputs,
        show_progress="hidden",
    )
    run_btn.click(
        on_run,
        inputs=[original_state, points_state, pending_state, outline_toggle, clean_toggle],
        outputs=result_outputs,
    )
    reset_btn.click(
        on_reset,
        inputs=[original_state, outline_toggle],
        outputs=result_outputs,
    )
    batch_toggle.change(
        on_toggle_batch,
        inputs=[
            batch_toggle, original_state, points_state, pending_state, outline_toggle, clean_toggle,
        ],
        outputs=[run_btn] + result_outputs,
    )
    outline_toggle.change(
        on_toggle_outline,
        inputs=[outline_toggle, original_state, points_state, pending_state],
        outputs=[image_display, mask_gallery],
        show_progress="hidden",
    )
    clean_toggle.change(
        on_toggle_clean,
        inputs=[clean_toggle, original_state, points_state, pending_state, outline_toggle],
        outputs=result_outputs,
    )


if __name__ == "__main__":
    demo.launch(footer_links=[])
