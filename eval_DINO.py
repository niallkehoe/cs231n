import cv2
from groundingdino.util.inference import load_model, predict
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt


# --- Load Grounding DINO ---

WEIGHTS_PATH = "/home/niall/weights/"
dino_config = "cs231n/config/GroundingDINO_SwinT_OGC.py"
dino_weights = "groundingdino_swint_ogc.pth"
dino_model = load_model(dino_config, WEIGHTS_PATH+dino_weights)


# --- Load Image ---

image_path = "example.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- Zero-shot Detection ---

prompt = "a cat, a chair, a laptop"
boxes, logits, phrases = predict(
    model=model,
    image=image_rgb,
    caption=prompt,
    box_threshold=0.35,
    text_threshold=0.25
)

# --- Visualize Detections ---

for box, label in zip(boxes, phrases):
    x0, y0, x1, y1 = map(int, box.tolist())
    cv2.rectangle(image_rgb, (x0, y0), (x1, y1), (255, 0, 0), 2)
    cv2.putText(image_rgb, label, (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

plt.imshow(image_rgb)
plt.title("Grounding DINO - Zero-shot Detection")
plt.axis("off")
plt.show()