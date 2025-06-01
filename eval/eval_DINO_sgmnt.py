"""
Grounded-SAM evaluation on a segmentation test-set.

Requirements
------------
pip install "torch>=2.1" torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install transformers==4.* pillow tqdm numpy opencv-python

For faster CPU preprocessing:
pip install safetensors accelerate

The script:
* detects the target object with Grounding DINO (open-vocabulary)
* converts the resulting boxes to pixel-accurate masks with Segment Anything
* aggregates metrics (Precision, Recall, F1, mIoU, Dice, AP@0.5, AP@0.5:0.95)
"""

import os, glob, argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,  # Grounding DINO
    SamProcessor, SamModel                               # Segment Anything
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities you already had
# ──────────────────────────────────────────────────────────────────────────────
# from helpers import (
#     load_segmentation_mask,
#     compute_segmentation_metrics,
#     plot_segmentation_to_image
# )
from aux_helpers import (
    load_segmentation_mask,
    compute_segmentation_metrics,
    plot_segmentation_to_image
)

# ──────────────────────────────────────────────────────────────────────────────
# 0) Load models once
# ──────────────────────────────────────────────────────────────────────────────
GD_MODEL_ID  = "IDEA-Research/grounding-dino-base"
SAM_MODEL_ID = "facebook/sam-vit-base"          # ~375 MB, fits most GPUs; change if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"▶ Loading Grounding DINO … {device}")
gd_processor = AutoProcessor.from_pretrained(GD_MODEL_ID)
gd_model     = AutoModelForZeroShotObjectDetection.from_pretrained(
                  GD_MODEL_ID).to(device).eval()

print("▶ Loading SAM …")
sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
sam_model     = SamModel.from_pretrained(SAM_MODEL_ID).to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# 1) Helper: Grounding DINO → boxes (absolute pixel coords)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def grounded_boxes(pil_img: Image.Image,
                   prompt: str,
                   box_thr: float,
                   text_thr: float):
    """Return (boxes tensor [N,4], scores tensor [N]) in pixel space."""
    text = prompt.lower().rstrip(".") + "."         # HF requirement
    inputs = gd_processor(images=pil_img, text=text, return_tensors="pt").to(device)
    outputs = gd_model(**inputs)

    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_thr,
        text_threshold=text_thr,
        target_sizes=[pil_img.size[::-1]]           # (H,W)
    )[0]

    if results["boxes"].numel() == 0:
        return torch.empty((0, 4)), torch.empty(0)

    return results["boxes"].cpu(), results["scores"].cpu()


# ──────────────────────────────────────────────────────────────────────────────
# 2) Helper: SAM → mask given boxes
# ──────────────────────────────────────────────────────────────────────────────
# ── improved SAM helper: boxes ➜ (H×W) union mask ────────────────────────────
@torch.no_grad()
def sam_union_mask(pil_img: Image.Image,
                   boxes_px: torch.Tensor) -> np.ndarray:
    """
    boxes_px : tensor [N,4] in absolute coords (x0,y0,x1,y1)
    returns  : np.uint8 mask (H_img, W_img) with values {0,1}
    """
    H_img, W_img = pil_img.size[1], pil_img.size[0]   # PIL gives (W,H)

    # No detections → blank mask
    if boxes_px.numel() == 0:
        return np.zeros((H_img, W_img), dtype=np.uint8)

    # 1. Prompt SAM
    sam_inputs = sam_processor(
        pil_img,
        input_boxes=[boxes_px.cpu().tolist()],         # B=1
        return_tensors="pt"
    ).to(device)

    sam_out = sam_model(**sam_inputs)

    # 2. Upscale logits to full-res
    masks = sam_processor.post_process_masks(
        sam_out.pred_masks,                           # [1,N,256,256]
        sam_inputs["original_sizes"],
        sam_inputs["reshaped_input_sizes"]
    )[0]                                              # [N, H_img', W_img']

    # 3. Threshold & union over instances
    union = (masks > 0.5).any(dim=0).cpu().numpy()    # bool or float32
    union = union.astype(np.uint8)                    # {0,1}

    # 4. Sanity-check shape
    union = np.squeeze(union)                         # drop (1, …) axes
    if union.ndim > 2:
        union = union[0]                              # take first plane
    if union.shape != (H_img, W_img):
        # Resize (nearest-neighbour) to *exact* image dims
        union = cv2.resize(union, (W_img, H_img),
                           interpolation=cv2.INTER_NEAREST)

    return union




# ──────────────────────────────────────────────────────────────────────────────
# 3) Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    all_preds, all_gts, all_pred_scores = [], [], []

    img_dir   = os.path.join(args.test_dir, "images")
    label_dir = os.path.join(args.test_dir, "labels")
    os.makedirs(args.output_dir, exist_ok=True)

    test_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"\n🖼  {len(test_files)} test images found")

    for img_idx, img_fp in tqdm(list(enumerate(test_files)), desc="Processing"):
        stem   = os.path.splitext(os.path.basename(img_fp))[0]
        lbl_fp = os.path.join(label_dir, stem + ".txt")

        pil_img = Image.open(img_fp).convert("RGB")

        # 3.1 detect boxes
        boxes, scores = grounded_boxes(
            pil_img,
            args.prompt,
            args.box_threshold,
            args.text_threshold
        )

        # 3.2 boxes → mask via SAM
        pred_mask = sam_union_mask(pil_img, boxes)

        # 3.3 load GT mask (expects same H×W as image)
        gt_mask = load_segmentation_mask(lbl_fp, pil_img.size)

        all_preds.append(pred_mask)
        all_gts.append(gt_mask)
        all_pred_scores.append(scores.max().item() if scores.numel() else 0.0)

        if args.verbose:
            vis = plot_segmentation_to_image(pil_img.copy(), pred_mask)
            vis.save(os.path.join(args.output_dir, f"{stem}_gsam.jpg"))

        # if img_idx >= 4:
        #     break

    # 4) Metrics
    miou, dice, precision, recall, f1, ap50, ap50_95 = compute_segmentation_metrics(
        all_preds, all_gts, all_pred_scores
    )

    print("\n[Grounded-SAM evaluation]")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Mean IoU       : {miou:.4f}")
    print(f"Dice Score     : {dice:.4f}")
    print(f"AP@0.5         : {ap50:.4f}")
    print(f"AP@0.5:0.95    : {ap50_95:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir",       type=str, required=True,
                        help="Folder containing images/ and labels/ sub-dirs")
    parser.add_argument("--prompt",         type=str, default="screen",
                        help="Object/category to detect (lower-cased automatically)")
    parser.add_argument("--box_threshold",  type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--output_dir",     type=str, default="out/grounded_sam")
    parser.add_argument("--verbose",        action="store_true",
                        help="Save visualisations for every frame")
    args = parser.parse_args()

    main(args)



"""
Detection (262 examples): (time: 5m 13s = 313s)
python eval_DINO_sgmnt.py \
    --test_dir ../datasets/segmentation-dataset/valid \
    --prompt "screen" \
    --output_dir out/dino/sgmnt \
    --box_threshold 0.3 \
    --text_threshold 0.25
  
Precision      : 0.8680
Recall         : 0.8211
F1 Score       : 0.8198
Mean IoU       : 0.7541
Dice Score     : 0.8198
AP@0.5         : 0.7721
AP@0.5:0.95    : 0.5850

Test (134 examples):
python eval_DINO_sgmnt.py \
    --test_dir ../datasets/segmentation-dataset/test \
    --prompt "screen" \
    --output_dir out/dino/sgmnt \
    --box_threshold 0.3 \
    --text_threshold 0.25

Precision      : 0.8236
Recall         : 0.8002
F1 Score       : 0.7903
Mean IoU       : 0.7091
Dice Score     : 0.7903
AP@0.5         : 0.7640
AP@0.5:0.95    : 0.4914
"""