import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from segment_anything import sam_model_registry, SamPredictor
from helpers import plot_boxes_to_image


# --- Config ---
HOME = "/home/niall/"
dino_config  = os.path.join(HOME, "cs231n/config/GroundingDINO_SwinT_OGC.py")
dino_weights = os.path.join(HOME, "weights/groundingdino_swint_ogc.pth")
sam_weights = os.path.join(HOME, "weights/sam_vit_h_4b8939.pth")
img_path = "citadel.png"

output_dir = "out"

text_prompt = "monitor"
box_threshold = 0.35
text_threshold = 0.25
token_spans = None

os.makedirs(output_dir, exist_ok=True)



def overlay_masks_on_image(image_pil, masks):
    image = np.array(image_pil).copy()
    for mask in masks:
        red = np.zeros_like(image)
        red[:, :, 0] = 255
        image = np.where(mask[..., None], red * 0.5 + image * 0.5, image)
    return Image.fromarray(image.astype(np.uint8))

# --- Utils ---
def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"x
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

# --- Load Models ---

# --- Load DINO ---
model = load_model(dino_config, dino_weights)

# --- Load Image ---
image_pil, image = load_image(img_path)

# visualize raw image
image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

# set the text_threshold to None if token_spans is set.
if token_spans is not None:
    text_threshold = None
    print("Using token_spans. Set the text_threshold to None.")

# run model
boxes_filt, pred_phrases = get_grounding_output(
    model, image, text_prompt, box_threshold, text_threshold, cpu_only=False, token_spans=eval(f"{token_spans}")
)

# visualize pred
size = image_pil.size
pred_dict = {
    "boxes": boxes_filt,
    "size": [size[1], size[0]],  # H,W
    "labels": pred_phrases,
}

# DINO Preds
image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
image_with_box.save(os.path.join(output_dir, "pred.jpg"))

print(f"Preds: {pred_dict}")

# --- SAM Segmentation ---


# --- Load SAM ---

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

# device = torch.device("cuda")  # will error if cuda isn't available
# sam = sam_model_registry["vit_b"](checkpoint=sam_weights)
# sam.to(device="cuda")

# print("made it")
# # double‐check
# for name, param in sam.named_parameters():
#     if param.device != device:
#         raise RuntimeError(f"Parameter {name} is on {param.device}, not {device}!")
# sam_predictor = SamPredictor(sam)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# sam = sam_model_registry["vit_b"](checkpoint=sam_weights)
# sam.to(device)

# # Ensure all parameters are on CUDA
# if not all(param.device.type == "cuda" for param in sam.parameters()):
#     print("ERROR: SAM model weights are not on CUDA. Exiting.")
#     sys.exit(1)

# print("predictor loading...")
# sam_predictor = SamPredictor(sam)

# masks = []
# if len(boxes_filt) > 0:
#     sam_predictor.set_image(np.array(image_pil))

#     for box in boxes_filt:
#         cx, cy, w, h = box.tolist()
#         x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
#         input_box = np.array([x0, y0, x1, y1])

#         masks_data = sam_predictor.predict(box=input_box, multimask_output=False)
#         mask = masks_data[0]
#         masks.append(mask)

#     # Overlay masks
#     image_with_masks = overlay_masks_on_image(image_pil, masks)
#     image_with_masks.save(os.path.join(output_dir, "segmentation.jpg"))



"""
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p /home/niall/weights/groundingdino_swint_ogc.pth \
-i /home/niall/cs231n/eval/citadel.png \
-o "/home/niall/" \
-t "monitor"
"""
