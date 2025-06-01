import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from helpers import (
    load_segmentation_mask,
    compute_segmentation_metrics,
    plot_segmentation_to_image
)

from groundingdino.util.inference import load_model, load_image, predict

to_tensor = T.ToTensor()       # 0-1 float32, C×H×W

def main(args):
    # ──────────────────────────────────────────────────────────────────────────
    # 1) Load the GroundingDINO model using config+weights
    # 2) Move it to CUDA if available, else keep on CPU
    # ──────────────────────────────────────────────────────────────────────────
    model = load_model(args.config, args.weights)
    device = torch.device("cpu")
    model = model.to(device)

    all_preds, all_gts, all_pred_scores = [], [], []

    test_img_dir = os.path.join(args.test_dir, "images")
    test_label_dir = os.path.join(args.test_dir, "labels")
    output_dir = args.output_dir or "out/dino/sgmnt"
    os.makedirs(output_dir, exist_ok=True)

    test_files = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    print(f"\nNumber of test images: {len(test_files)}")

    for i, img_fp in tqdm(enumerate(test_files), desc="Processing images"):
        basename = os.path.splitext(os.path.basename(img_fp))[0]
        lbl_fp = os.path.join(test_label_dir, basename + ".txt")

        pil_img = Image.open(img_fp).convert("RGB")
        # img_tensor = to_tensor(pil_img).to(device)         # ← NEW: tensor on cpu/gpu
                                                       # (shape 3×H×W, values 0-1)
        # image_source  -> the untouched PIL image (good for visualisation)
        # image         -> a C×H×W float32 tensor, resized + normalised the way DINO expects
        image_source, image = load_image(img_fp)      # 👈 one-liner that does it all
        image = image.to(device)
                                                       
        gt_mask = load_segmentation_mask(lbl_fp, pil_img.size)
        all_gts.append(gt_mask)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption="screen",
            box_threshold=0.3,
            text_threshold=0.25,
            device="cpu"
        )

        pred_mask = np.zeros(gt_mask.shape, dtype=np.uint8)

        if boxes is not None and len(boxes) > 0:
            # Simple bounding-box-to-mask approximation
            for box in boxes:
                x1, y1, x2, y2 = box.int().cpu().numpy()
                pred_mask[y1:y2, x1:x2] = 1
            score = float(torch.max(logits).item())
        else:
            score = 0.0

        all_preds.append(pred_mask)
        all_pred_scores.append(score)

        # print(f"pred_mask: {pred_mask.shape}")
        # print(pred_mask)

        if args.verbose:
            vis_img = plot_segmentation_to_image(pil_img.copy(), pred_mask)
            vis_img.save(os.path.join(output_dir, f"{basename}_dino.jpg"))

        if i > 1:
            break

    miou, dice, precision, recall, f1, ap50, ap50_95 = compute_segmentation_metrics(
        all_preds, all_gts, all_pred_scores
    )

    print(f"\n[DINO Evaluation Metrics]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Dice Score: {dice:.4f}")
    print(f"AP@0.5: {ap50:.4f}")
    print(f"AP@0.5:0.95: {ap50_95:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--weights", type=str, default="../weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="out/dino/sgmnt")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)



"""

Val (158 examples): (time: )

python eval_DINO_sgmnt.py \
    --config ../config/GroundingDINO_SwinT_OGC.py \
    --weights ../../weights/groundingdino_swint_ogc.pth \
    --test_dir ../datasets/segmentation-dataset/test \
    --output_dir out/dino/sgmnt \
    --verbose

Precision: 0.7493
Recall: 0.8532
F1 Score: 0.7666
Mean IoU: 0.6997
Dice Score: 0.7666
AP@0.5: 0.6926
AP@0.5:0.95: 0.5123

Test (81 examples):
python eval_DINO_sgmnt.py \
    --model_path ../train/runs/segment/medium/weights/best.pt \
    --test_dir ../datasets/segmentation-dataset/test \
    --output_dir out/dino/sgmnt

Precision: 0.7630
Recall: 0.8783
F1 Score: 0.7906
Mean IoU: 0.7263
Dice Score: 0.7906
AP@0.5: 0.6917
AP@0.5:0.95: 0.5576
"""