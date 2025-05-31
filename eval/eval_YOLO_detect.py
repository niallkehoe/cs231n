import os
import glob
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

from helpers import plot_boxes_to_image, load_yolo_boxes, compute_metrics, compute_ap_metrics

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
IOU_THR = 0.5  # for F1 computation

def main(args):
    model = YOLO(args.model_path)
    metrics = {"tp": 0, "fp": 0, "fn": 0, "ious": []}

    all_pred_boxes = []
    all_gt_boxes = []

    test_img_dir = os.path.join(args.test_dir, "images")
    test_label_dir = os.path.join(args.test_dir, "labels")
    output_dir = args.output_dir or "out/yolo/"
    os.makedirs(output_dir, exist_ok=True)

    test_files = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    print(f"\nNumber of test images: {len(test_files)}")

    for i, img_fp in tqdm(enumerate(test_files), total=len(test_files), desc="Processing images"):
        basename = os.path.splitext(os.path.basename(img_fp))[0]
        lbl_fp = os.path.join(test_label_dir, basename + ".txt")

        pil = Image.open(img_fp).convert("RGB")
        W, H = pil.size

        gt_boxes = load_yolo_boxes(lbl_fp, W, H, class_id=0)
        all_gt_boxes.append(gt_boxes)

        results = model(img_fp, stream=True, verbose=False)

        preds, scores = [], []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != 0:
                    continue

                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                preds.append(xyxy)
                scores.append(conf)

            if args.verbose:
                pred_dict = {
                    "boxes": torch.tensor(preds),
                    "size": [H, W],
                    "labels": [f"monitor ({box.conf[0].item():.2f})" for box in result.boxes if int(box.cls[0].item()) == 0]
                }
                img_with_box = plot_boxes_to_image(pil, pred_dict, boxes_are_normalized=False)[0]
                out_fp = os.path.join(output_dir, f"{basename}_pred.jpg")
                img_with_box.save(out_fp)
                print(f"[verbose] saved {out_fp}")

        preds = torch.tensor(preds) if preds else torch.zeros((0, 4))
        all_pred_boxes.append(preds)

        if preds.numel() > 0 and gt_boxes.numel() > 0:
            from groundingdino.util.box_ops import box_iou
            iou_mat, _ = box_iou(preds, gt_boxes)
        else:
            iou_mat = torch.zeros((len(preds), len(gt_boxes)))

        matched_gt = set()
        for pi in range(iou_mat.shape[0]):
            if iou_mat.shape[1] == 0:
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

    p, r, f1, mean_iou = compute_metrics(metrics["ious"], metrics["tp"], metrics["fp"], metrics["fn"])
    print(f"Precision: {p:.4f}\nRecall:    {r:.4f}\nF1 Score:  {f1:.4f}\nMean IoU:  {mean_iou:.4f}")

    ap50, ap5095, mean_iou = compute_ap_metrics(all_pred_boxes, all_gt_boxes)
    print(f"AP@0.5: {ap50:.4f}")
    print(f"AP@0.5:0.95: {ap5095:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to YOLO model")
    parser.add_argument("--test_dir", type=str, help="Directory with test images and labels")
    parser.add_argument("--output_dir", type=str, help="Directory to save visual outputs")
    parser.add_argument("--verbose", action="store_true", help="Save each image with predicted boxes into output dir")
    args = parser.parse_args()
    main(args)


"""

python evaluate_yolo.py \
    --model_path ../train/runs/detect/medium/weights/best.pt \
    --test_dir ../datasets/detection-dataset/test \
    --output_dir out/yolo/detect \
    --verbose

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