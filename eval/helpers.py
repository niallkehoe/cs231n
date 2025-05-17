import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# --- Plotting function from other script ---
# def plot_boxes_to_image(image_pil, tgt):
#     H, W = tgt["size"]
#     boxes = tgt["boxes"]
#     labels = tgt["labels"]
#     assert len(boxes) == len(labels), "boxes and labels must have same length"

#     draw = ImageDraw.Draw(image_pil)
#     mask = Image.new("L", image_pil.size, 0)
#     mask_draw = ImageDraw.Draw(mask)

#     # draw boxes and masks
#     for box, label in zip(boxes, labels):
#         # from 0..1 to 0..W, 0..H
#         box = box * torch.Tensor([W, H, W, H])
#         # from xywh to xyxy
#         box[:2] -= box[2:] / 2
#         box[2:] += box[:2]
#         # random color
#         color = tuple(np.random.randint(0, 255, size=3).tolist())
#         # draw
#         x0, y0, x1, y1 = box
#         x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

#         draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
#         # draw.text((x0, y0), str(label), fill=color)

#         font = ImageFont.load_default()
#         if hasattr(font, "getbbox"):
#             bbox = draw.textbbox((x0, y0), str(label), font)
#         else:
#             w, h = draw.textsize(str(label), font)
#             bbox = (x0, y0, w + x0, y0 + h)
#         # bbox = draw.textbbox((x0, y0), str(label))
#         draw.rectangle(bbox, fill=color)
#         draw.text((x0, y0), str(label), fill="white")

#         mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

#     return image_pil, mask

def plot_boxes_to_image(image_pil, tgt, boxes_are_normalized=False):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        if boxes_are_normalized:
            # convert from normalized cx, cy, w, h to absolute xyxy
            box = box * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2  # cx,cy -> top-left
            box[2:] += box[:2]      # w,h -> bottom-right
        # else: box is already in absolute xyxy format

        x0, y0, x1, y1 = map(int, box)

        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        # draw label
        font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font=font)
        else:
            w, h = draw.textsize(str(label), font=font)
            bbox = (x0, y0, x0 + w, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask



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


def yolo_to_xyxy(box, img_w, img_h):
    # box = [x_center_norm, y_center_norm, w_norm, h_norm]
    x_c, y_c, w, h = box
    x0 = (x_c - w/2) * img_w
    y0 = (y_c - h/2) * img_h
    x1 = (x_c + w/2) * img_w
    y1 = (y_c + h/2) * img_h
    return [x0, y0, x1, y1]

def load_yolo_boxes(label_path, img_w, img_h, class_id=0):
    """
    Reads either YOLO bboxes (xc, yc, w, h) *or* polygon segmentations
    (x1, y1, x2, y2, ..., xn, yn), all normalized [0..1]. Returns a
    Tensor of [x0, y0, x1, y1] in absolute pixel coords.
    """
    gt = []
    with open(label_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if not parts:
                continue
            cls = int(parts[0])
            if cls != class_id:
                continue

            coords = parts[1:]
            # standard YOLO box
            if len(coords) == 4:
                gt.append(yolo_to_xyxy(coords, img_w, img_h))

            # polygon segmentation: x1,y1, x2,y2, ...
            elif len(coords) > 4 and len(coords) % 2 == 0:
                xs = [coords[i] * img_w   for i in range(0, len(coords), 2)]
                ys = [coords[i] * img_h   for i in range(1, len(coords), 2)]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                gt.append([x0, y0, x1, y1])

            else:
                # unexpected format
                print(f"⚠️  Skipping invalid label line in {label_path!r}: {line.strip()}")

    if not gt:
        return torch.zeros((0, 4))
    return torch.tensor(gt, dtype=torch.float32)


def compute_metrics(all_iou_scores, tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp)>0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn)>0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision+recall)>0 else 0.0
    mean_iou  = np.mean(all_iou_scores) if all_iou_scores else 0.0
    return precision, recall, f1, mean_iou