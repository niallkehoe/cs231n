import os
import glob
import numpy as np
import torch
from PIL import Image
import argparse
from tqdm import tqdm

# Your existing imports
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.box_ops import box_iou  # returns (iou_matrix, union_matrix)
from helpers import plot_boxes_to_image, load_yolo_boxes, yolo_to_xyxy, compute_metrics

# --- Config ---
HOME = "/home/niall/"
dino_config  = os.path.join(HOME, "cs231n/config/GroundingDINO_SwinT_OGC.py")
dino_weights = os.path.join(HOME, "weights/groundingdino_swint_ogc.pth")

TEST_IMG_DIR   = HOME + "cs231n/datasets/detection-dataset/test/images"
TEST_LABEL_DIR = HOME + "cs231n/datasets/detection-dataset/test/labels"
OUTPUT_DIR     = "out/dino/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT         = "monitor"
BOX_THR        = 0.35
TEXT_THR       = 0.25
IOU_THR        = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(config_path, ckpt_path, device=DEVICE):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    sd = torch.load(ckpt_path, map_location="cpu")["model"]
    model.load_state_dict(clean_state_dict(sd), strict=False)
    return model.eval().to(device)


def load_image_and_transform(path):
    pil = Image.open(path).convert("RGB")
    tf = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])
    img, _ = tf(pil, None)
    return pil, img.to(DEVICE)


# def get_grounding_boxes(model, image_tensor, prompt, box_thr, text_thr):
#     # identical to your get_grounding_output, but returns raw boxes [cx,cy,w,h] in 0–1
#     prompt = prompt.lower().strip()
#     if not prompt.endswith("."):
#         prompt += "."
#     with torch.no_grad():
#         out = model(image_tensor[None], captions=[prompt])
#     logits = out["pred_logits"].sigmoid()[0]  # (nq,256)
#     boxes  = out["pred_boxes"][0]             # (nq,4)
#     # filter by box threshold
#     mask   = logits.max(dim=1)[0] > box_thr
#     boxes  = boxes[mask].cpu()
#     logits = logits[mask].cpu()
#     # filter further by text threshold
#     keep = (logits > text_thr).any(dim=1)
#     return boxes[keep]  # tensor [N,4]

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases



def main(args):
    model = load_model(dino_config, dino_weights)
    metrics = {"tp":0, "fp":0, "fn":0, "ious":[]}
    test_files = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    print(f"\nNumber of test images: {len(test_files)}")

    for i, img_fp in tqdm(enumerate(test_files), total=len(test_files), desc="Processing images"):
        basename = os.path.splitext(os.path.basename(img_fp))[0]
        lbl_fp   = os.path.join(TEST_LABEL_DIR, basename + ".txt")

        pil, img_t = load_image_and_transform(img_fp)
        W, H      = pil.size
        gt_boxes  = load_yolo_boxes(lbl_fp, W, H, class_id=0)  # only “monitor” class

        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            model, img_t, PROMPT, BOX_THR, TEXT_THR
        )
        # print(f"Pred phrases: {pred_phrases}")

        if args.verbose: # draw and save
            # reuse your plotting function to overlay boxes
            pred_dict = {
                "boxes": boxes_filt,
                "size": [H, W],
                "labels": pred_phrases
            }
            # print(f"Preds: {pred_dict}")
            img_with_box = plot_boxes_to_image(pil, pred_dict)[0]
            out_fp = os.path.join(OUTPUT_DIR, f"{basename}_pred.jpg")
            img_with_box.save(out_fp)
            print(f"[verbose] saved {out_fp}")

        # ——— convert filtered grounding boxes to absolute xyxy ———
        preds = []
        for b in boxes_filt:
            cx, cy, w, h = b.tolist()
            preds.append(yolo_to_xyxy([cx, cy, w, h], W, H))
        preds = torch.tensor(preds) if preds else torch.zeros((0,4))

        # load GT boxes
        gt_boxes = load_yolo_boxes(lbl_fp, W, H, class_id=0)

        # ——— compute IoU matrix ———
        if preds.numel()>0 and gt_boxes.numel()>0:
            iou_mat, _ = box_iou(preds, gt_boxes)
        else:
            iou_mat = torch.zeros((len(preds), len(gt_boxes)))

        # ——— greedy one-to-one matching ———
        matched_gt = set()
        for pi in range(iou_mat.shape[0]):
            best_gt = int(torch.argmax(iou_mat[pi]))
            best_iou = float(iou_mat[pi, best_gt])
            if best_iou >= IOU_THR and best_gt not in matched_gt:
                metrics["tp"] += 1
                metrics["ious"].append(best_iou)
                matched_gt.add(best_gt)
            else:
                metrics["fp"] += 1

        # any ground‐truth boxes left unmatched are false negatives
        metrics["fn"] += (len(gt_boxes) - len(matched_gt))

        # if i > 50:
        #     break

    p, r, f1, mean_iou = compute_metrics(metrics["ious"], metrics["tp"], metrics["fp"], metrics["fn"])
    print(f"Precision: {p:.4f}\nRecall:    {r:.4f}\nF1 Score:  {f1:.4f}\nMean IoU:  {mean_iou:.4f}")

    
if __name__ == "__main__":
    # parse command‐line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="save each image with predicted boxes into out/")
    args = parser.parse_args()
    main(args)

