# inference/yolo_infer.py

from ultralytics import YOLO
import numpy as np

def load_yolo_model(weight_path):
    model = YOLO(weight_path)
    return model

def run_yolo(model, image):
    results = model(image)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)
    class_names = results.names

    return boxes, labels, scores, class_names