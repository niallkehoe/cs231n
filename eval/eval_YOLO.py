import os
import glob
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import logging
import numpy as np

from helpers import plot_boxes_to_image, load_yolo_boxes, yolo_to_xyxy, compute_metrics, compute_ap_metrics, prepare_boxes_for_ap

# --- Config ---
HOME = "/home/niall/cs231n/"

# YOLO_MODEL_PATH = os.path.join(HOME, "train/runs/detect/small/weights/best.pt")
YOLO_MODEL_PATH = os.path.join(HOME, "train/runs/detect/medium/weights/best.pt")

TEST_DIR = HOME + "datasets/detection-dataset/valid"
TEST_DIR = HOME + "datasets/detection-dataset/test"
TEST_IMG_DIR = TEST_DIR + "/images"
TEST_LABEL_DIR = TEST_DIR + "/labels"
# TEST_IMG_DIR = HOME + "eval/input"

OUTPUT_DIR = "out/yolo/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
IOU_THR = 0.5  # for F1 computation

# This must be implemented elsewhere already:
# - load_yolo_boxes(label_path, width, height, class_id)
# - yolo_to_xyxy(box, W, H)
# - compute_metrics(ious, tp, fp, fn)


def main(args):
    model = YOLO(YOLO_MODEL_PATH)
    metrics = {"tp":0, "fp":0, "fn":0, "ious":[]}

    # For AP50 / mAP
    all_pred_boxes = []
    all_gt_boxes = []

    test_files = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    print(f"\nNumber of test images: {len(test_files)}")

    for i, img_fp in tqdm(enumerate(test_files), total=len(test_files), desc="Processing images"):
        basename = os.path.splitext(os.path.basename(img_fp))[0]
        lbl_fp = os.path.join(TEST_LABEL_DIR, basename + ".txt")

        pil = Image.open(img_fp).convert("RGB")
        W, H = pil.size

        # Load GT boxes
        gt_boxes = load_yolo_boxes(lbl_fp, W, H, class_id=0)  # only “monitor” class
        all_gt_boxes.append(gt_boxes)

        # YOLO inference
        results = model(img_fp, stream=True, verbose=False)

        preds, scores = [], []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != 0:  # filter only "monitor" class
                    continue

                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].item())
                preds.append(xyxy)
                scores.append(conf)

            if args.verbose:
                pred_dict = {
                    "boxes": torch.tensor(preds),
                    "size": [H, W],
                    "labels": [f"monitor ({box.conf[0].item():.2f})" for box in result.boxes if int(box.cls[0].item()) == 0]
                }
                print(f"Preds: {pred_dict}")
                img_with_box = plot_boxes_to_image(pil, pred_dict, boxes_are_normalized=False)[0]
                out_fp = os.path.join(OUTPUT_DIR, f"{basename}_pred.jpg")
                img_with_box.save(out_fp)
                print(f"[verbose] saved {out_fp}")

        preds = torch.tensor(preds) if preds else torch.zeros((0, 4))
        all_pred_boxes.append(preds)

        # Compute IoU matrix
        if preds.numel() > 0 and gt_boxes.numel() > 0:
            from groundingdino.util.box_ops import box_iou
            iou_mat, _ = box_iou(preds, gt_boxes)
        else:
            iou_mat = torch.zeros((len(preds), len(gt_boxes)))

        # Greedy one-to-one matching
        matched_gt = set()
        for pi in range(iou_mat.shape[0]):
            if iou_mat.shape[1] == 0:
                # No GT boxes to match with; all preds are false positives
                metrics["fp"] += 1
                continue

            best_gt = int(torch.argmax(iou_mat[pi]))
            best_iou = float(iou_mat[pi, best_gt])
            if best_iou >= IOU_THR and best_gt not in matched_gt:
                metrics["tp"] += 1
                metrics["ious"].append(best_iou)
                matched_gt.add(best_gt)
            else:
                metrics["fp"] += 1

        metrics["fn"] += (len(gt_boxes) - len(matched_gt))

        # if i > 5:
        #     break

    # Final scores
    p, r, f1, mean_iou = compute_metrics(metrics["ious"], metrics["tp"], metrics["fp"], metrics["fn"])
    print(f"Precision: {p:.4f}\nRecall:    {r:.4f}\nF1 Score:  {f1:.4f}\nMean IoU:  {mean_iou:.4f}")

    # Compute AP50 / mAP
    ap50, ap5095, mean_iou = compute_ap_metrics(all_pred_boxes, all_gt_boxes)
    print(f"AP@0.5: {ap50:.4f}")
    print(f"AP@0.5:0.95: {ap5095:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Save each image with predicted boxes into out/")
    args = parser.parse_args()
    main(args)


"""

Val: (time: 20s)
Precision: 0.8418
Recall:    0.9110
F1 Score:  0.8750
Mean IoU:  0.8963
AP@0.5: 0.8418
AP@0.5:0.95: 0.7099
Mean IoU: 0.9116


Test:

Precision: 0.8333
Recall:    0.9028
F1 Score:  0.8667
Mean IoU:  0.9009
AP@0.5: 0.8333
AP@0.5:0.95: 0.7083
Mean IoU: 0.9131
"""