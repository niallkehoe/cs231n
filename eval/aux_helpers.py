import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import cv2

# --- Segmentation ---


def load_segmentation_mask(lbl_fp: str, img_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a YOLO-v8 segmentation label (.txt with polygon coords) into a binary mask.
    Args:
        lbl_fp (str): path to the .txt label file
        img_size (Tuple[int, int]): (width, height) of the corresponding image
    Returns:
        np.ndarray: binary mask of shape (H, W) with values 0/1
    """
    w, h = img_size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not os.path.exists(lbl_fp):
        return mask  # no objects => empty mask
    
    with open(lbl_fp, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 6:  # need at least 3 (x,y) pairs
                continue
            poly = np.array(parts[1:]).reshape(-1, 2)  # drop class id
            pts_int = np.round(poly * [w, h]).astype(np.int32)
            cv2.fillPoly(mask, [pts_int], 1)
    
    return mask

def compute_iou_at_threshold(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """Compute IoU at a specific threshold."""
    pred_binary = pred >= threshold
    gt_binary = gt.astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def compute_average_precision(pred_scores: List[float], gt_masks: List[np.ndarray], 
                            pred_masks: List[np.ndarray], iou_threshold: float) -> float:
    """
    Compute Average Precision at a specific IoU threshold.
    """
    # Create list of (score, is_positive) pairs
    detections = []
    
    for i, (score, pred_mask, gt_mask) in enumerate(zip(pred_scores, pred_masks, gt_masks)):
        # Check if there's a ground truth object
        has_gt = gt_mask.sum() > 0
        
        if pred_mask.sum() > 0:  # Only consider non-empty predictions
            iou = compute_iou_at_threshold(pred_mask.astype(float), gt_mask, 0.5)
            is_positive = iou >= iou_threshold and has_gt
            detections.append((score, is_positive, has_gt))
    
    if not detections:
        return 0.0
    
    # Sort by confidence score (descending)
    detections.sort(key=lambda x: x[0], reverse=True)
    
    # Compute precision and recall at each threshold
    tp = 0
    fp = 0
    total_positives = sum(1 for _, _, has_gt in detections if has_gt)
    
    if total_positives == 0:
        return 0.0
    
    precisions = []
    recalls = []
    
    for score, is_positive, has_gt in detections:
        if is_positive:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_positives if total_positives > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Compute AP using the 11-point interpolation method
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        # Find precisions for recalls >= t
        p_max = 0
        for i, r in enumerate(recalls):
            if r >= t:
                p_max = max(p_max, precisions[i])
        ap += p_max / 11
    
    return ap

def compute_segmentation_metrics(preds: List[np.ndarray], gts: List[np.ndarray], 
                               pred_scores: List[float] = None) -> Tuple[float, float, float, float, float, float, float]:
    """
    Computes comprehensive segmentation metrics including AP.
    Args:
        preds (List[np.ndarray]): List of predicted binary masks (0 or 1), shape (H, W).
        gts (List[np.ndarray]): List of ground truth binary masks (0 or 1), shape (H, W).
        pred_scores (List[float]): List of confidence scores for predictions.
    Returns:
        Tuple: mean IoU, mean Dice score, mean precision, mean recall, mean F1, AP@0.5, AP@0.5:0.95
    """
    eps = 1e-6
    iou_list, dice_list, precision_list, recall_list, f1_list = [], [], [], [], []
    
    # Basic metrics computation
    for pred, gt in zip(preds, gts):
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        
        iou = intersection / (union + eps)
        dice = 2 * intersection / (pred.sum() + gt.sum() + eps)
        precision = intersection / (pred.sum() + eps)
        recall = intersection / (gt.sum() + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        iou_list.append(iou)
        dice_list.append(dice)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # AP computation
    if pred_scores is None:
        pred_scores = [1.0] * len(preds)  # Default confidence
    
    # AP@0.5
    ap50 = compute_average_precision(pred_scores, gts, preds, 0.5)
    
    # AP@0.5:0.95 (average over IoU thresholds from 0.5 to 0.95 with step 0.05)
    ap_scores = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        ap = compute_average_precision(pred_scores, gts, preds, iou_thresh)
        ap_scores.append(ap)
    ap50_95 = np.mean(ap_scores)
    
    return (
        np.mean(iou_list),      # Mean IoU
        np.mean(dice_list),     # Mean Dice
        np.mean(precision_list), # Mean Precision
        np.mean(recall_list),   # Mean Recall
        np.mean(f1_list),       # Mean F1
        ap50,                   # AP@0.5
        ap50_95                 # AP@0.5:0.95
    )

# def plot_segmentation_to_image(img, mask):
#     """Plot segmentation mask overlay on image."""
#     overlay = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
#     red = Image.new("RGB", img.size, (255, 0, 0))
#     img.paste(red, mask=overlay)
#     return img

def plot_segmentation_to_image(img, mask):
    """Plot segmentation mask overlay on image."""
    # 1) Convert mask to a PIL “L” image (values 0–255)
    overlay = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")

    # 2) If overlay size ≠ img size, resize to match (use NEAREST to keep hard edges)
    if overlay.size != img.size:
        overlay = overlay.resize(img.size, resample=Image.NEAREST)

    # 3) Create a red image of the same size
    red = Image.new("RGB", img.size, (255, 0, 0))

    # 4) Paste the red over img, using overlay as the mask
    img.paste(red, mask=overlay)
    return img