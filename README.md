# ScreenShield: Computer Monitor Tracking and Blurring for Video

**ScreenShield** is a real-time video processing pipeline that detects, segments, tracks, and inpaints computer monitor screens in video. It's optimized for low-latency use cases like livestreams, video conferencing, or on-device video capture where sensitive screen content must be obscured dynamically.

---

## 🛡️ Key Features

- **Real-Time Screen Obfuscation** (~0.15 sec/frame)
- **YOLOv11-based Detection & Segmentation**
- **Cutie-powered Mask Propagation**
- **Configurable Inpainting (Gaussian Blur, Logo Overlay, Border)**
- **Supports Stationary and Moving Cameras**

---

## 🎥 Demo Videos

- 🔍 [Original input clip](https://youtu.be/QP8muGb9pRc)  
- 🟡 [Blurred screen output](https://youtu.be/-lSK1XjPA-g)  
- 🎯 [Blur + logo overlay output](https://youtu.be/nOqEOoKYxv0)  

---

## 🧠 Pipeline Overview

The ScreenShield pipeline consists of four key stages:

1. **Detection**  
   A fine-tuned YOLOv11 model is run on each input frame to detect computer monitors using bounding boxes. Once a monitor is detected, the system proceeds to segmentation.

2. **Segmentation**  
   The frame is passed to a YOLOv11-based segmentation model to generate a pixel-level mask of each screen.

3. **Tracking**  
   Cutie (a video object segmentation model) tracks the segmented masks across subsequent frames to avoid redundant segmentation. If the number of screen instances changes, the system re-runs segmentation.

4. **Inpainting**  
   Each mask region is blurred using a Gaussian blur. Optionally, users can overlay a custom logo or a white border on each screen region.


## 📊 Performance

| Task              | Model                | Speed (T4 GPU)     | F1 Score | AP@0.5:0.95 |
|-------------------|----------------------|--------------------|----------|-------------|
| Detection         | YOLOv11 (finetuned)  | 0.02s / frame      | 0.87     | 0.71        |
| Segmentation      | YOLOv11-Seg (finetuned) | 0.15s / frame   | 0.78     | 0.60        |

---

## 📂 Datasets

- **Detection**
  - Roboflow Computer Monitor Dataset
  - Roboflow TV/PC/Monitor Dataset
  - Roboflow Office Monitor Dataset
- **Segmentation**
  - Roboflow Screen Segmentation
  - Custom YouTube dataset (hand-labeled)

## 🧪 Conda Environments

The `envs/` folder contains pre-defined environment YAML files to support different stages of the pipeline. You can use these to quickly reproduce our setup:

| File          | Purpose                                                                 |
|---------------|-------------------------------------------------------------------------|
| `train.yml`   | Environment for training the YOLOv11 detection and segmentation models. |
| `infer.yml`   | Lightweight environment for running inference on new videos.            |
| `eval.yml`    | Environment for benchmarking and evaluating model performance.          |

### 🔄 Usage

You can create any of these environments with:

```bash
mamba env create -f envs/train.yml
mamba env create -f envs/infer.yml
mamba env create -f envs/eval.yml
```

---

## 🧑‍💻 Authors

- Niall Kehoe – nkehoe@stanford.edu
- Aniket Mahajan – aniketm@stanford.edu
