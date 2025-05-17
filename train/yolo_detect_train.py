from ultralytics import YOLO
import os

# kill switch to kill the gcp instance after training
KILL_SWITCH = True

# TRAIN_DATA_PATH = "../datasets/detection-dataset/data.yaml"

# # Load a COCO-pretrained YOLO11n model
# model = YOLO("yolo11n.pt")

# print(f"Training model on {TRAIN_DATA_PATH}")
# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data=TRAIN_DATA_PATH, epochs=100, imgsz=640)

# Load a model
# TRAIN_MODEL_PATH = "runs/detect/train2/weights/last.pt"
TRAIN_DATA_PATH = "runs/detect/train/weights/last.pt"
model = YOLO(TRAIN_MODEL_PATH)  # load a partially trained model

# Resume training
results = model.train(resume=True)

if KILL_SWITCH:
    # kill the gcp instance, run `sudo shutdown -h now`
    print("KILL_SWITCH is ON. Shutting down the instance...")
    os.system("sudo shutdown -h now")

