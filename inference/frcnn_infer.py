import torch
import torchvision
from torchvision.transforms import functional as F

# 11 object classes + 1 background = 12
NUM_CLASSES = 12  

# Class names must match training exactly
CLASS_NAMES = [
    "background",          # 0
    "pedestrian",          # 1
    "people",              # 2
    "bicycle",             # 3
    "car",                 # 4
    "van",                 # 5
    "truck",               # 6
    "tricycle",            # 7
    "awning-tricycle",     # 8
    "bus",                 # 9
    "motor",               # 10
    "others"               # 11
]


def load_frcnn_model(weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, NUM_CLASSES
    )

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    print("✅ Faster R-CNN model loaded successfully")
    return model


def run_frcnn(model, image):
    device = next(model.parameters()).device

    # Convert PIL image to tensor
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()
    labels = outputs["labels"].detach().cpu().numpy()

    # 🔥 Debug Information
    print("------ Faster R-CNN Debug ------")
    print("Total raw detections:", len(scores))

    if len(scores) > 0:
        print("Max score:", scores.max())
        print("Min score:", scores.min())
    else:
        print("No detections returned by model.")

    # Remove background class (label = 0)
    valid_indices = labels != 0

    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]

    print("Detections after removing background:", len(scores))
    print("--------------------------------")

    return boxes, labels, scores, CLASS_NAMES