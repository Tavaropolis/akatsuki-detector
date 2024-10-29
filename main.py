import cv2
from ultralytics import YOLO
from roboflow import Roboflow

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data="BEPPE-2/data.yaml", epochs=1, imgsz=640)

model = YOLO("./best.pt")

results = model.track(source= "video2.webm", show = True, conf=0.6,  save = True)
