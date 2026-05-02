# utils/draw_boxes.py

import cv2
from utils.label_mapping import map_label

def draw_boxes(image, boxes, labels, scores, class_names, conf_threshold=0.4):
    img = image.copy()

    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        mapped_label = map_label(class_names[label])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{mapped_label} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return img