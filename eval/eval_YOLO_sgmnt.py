import os
import glob
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from helpers import load_segmentation_mask, compute_segmentation_metrics, plot_segmentation_to_image

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

def main(args):
    model = YOLO(args.model_path)
    all_preds = []
    all_gts = []
    all_pred_scores = []  # Store confidence scores for AP calculation
    
    test_img_dir = os.path.join(args.test_dir, "images")
    test_label_dir = os.path.join(args.test_dir, "labels")
    output_dir = args.output_dir or "out/yolo_seg/"
    os.makedirs(output_dir, exist_ok=True)
    
    test_files = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    print(f"\nNumber of test images: {len(test_files)}")
    
    for i, img_fp in tqdm(enumerate(test_files), total=len(test_files), desc="Processing images"):
        basename = os.path.splitext(os.path.basename(img_fp))[0]
        lbl_fp = os.path.join(test_label_dir, basename + ".txt")
        
        img = Image.open(img_fp).convert("RGB")
        gt_mask = load_segmentation_mask(lbl_fp, img.size)  # <-- pass (W,H)
        all_gts.append(gt_mask)
        
        results = model(img_fp, stream=False, verbose=False)[0]
        
        if results.masks is not None:
            # Shape: (N, H, W)
            pred_masks = results.masks.data.cpu().numpy()
            pred_mask = np.any(pred_masks.astype(bool), axis=0).astype(np.uint8)
            
            # Get confidence scores (use max confidence if multiple masks)
            if hasattr(results, 'boxes') and results.boxes is not None:
                pred_score = float(results.boxes.conf.max().cpu().numpy())
            else:
                pred_score = 1.0  # Default confidence if not available
        else:
            pred_mask = np.zeros_like(gt_mask)
            pred_score = 0.0
        
        # Resize pred_mask to match gt_mask if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask = Image.fromarray(pred_mask.astype(np.uint8))
            pred_mask = pred_mask.resize(gt_mask.shape[::-1], resample=Image.NEAREST)
            pred_mask = np.array(pred_mask)
        
        all_preds.append(pred_mask)
        all_pred_scores.append(pred_score)
        
        if args.verbose:
            vis_img = plot_segmentation_to_image(img, pred_mask)
            out_fp = os.path.join(output_dir, f"{basename}_seg.jpg")
            vis_img.save(out_fp)

        # break
    
    # Compute enhanced metrics
    miou, dice, precision, recall, f1, ap50, ap50_95 = compute_segmentation_metrics(
        all_preds, all_gts, all_pred_scores
    )
    
    print(f"\nEvaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Dice Score: {dice:.4f}")
    print(f"AP@0.5: {ap50:.4f}")
    print(f"AP@0.5:0.95: {ap50_95:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to YOLO segmentation model")
    parser.add_argument("--test_dir", type=str, help="Directory with test images and masks")
    parser.add_argument("--output_dir", type=str, help="Directory to save visual outputs")
    parser.add_argument("--verbose", action="store_true", help="Save predicted masks to output dir")
    args = parser.parse_args()
    main(args)


"""

Val (158 examples): (time: 02:17)
python eval_YOLO_sgmnt.py \
    --model_path ../train/runs/segment/medium/weights/best.pt \
    --test_dir ../datasets/segmentation-dataset/valid \
    --output_dir out/yolo/sgmnt

Precision: 0.7493
Recall: 0.8532
F1 Score: 0.7666
Mean IoU: 0.6997
Dice Score: 0.7666
AP@0.5: 0.6926
AP@0.5:0.95: 0.5123

Test (81 examples):
python eval_YOLO_sgmnt.py \
    --model_path ../train/runs/segment/medium/weights/best.pt \
    --test_dir ../datasets/segmentation-dataset/test \
    --output_dir out/yolo/sgmnt

Precision: 0.7630
Recall: 0.8783
F1 Score: 0.7906
Mean IoU: 0.7263
Dice Score: 0.7906
AP@0.5: 0.6917
AP@0.5:0.95: 0.5576
"""