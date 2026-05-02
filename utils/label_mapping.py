# utils/label_mapping.py

VEHICLE_CLASSES = [
    "car",
    "van",
    "truck",
    "bus",
    "tricycle",
    "awning-tricycle",
    "motor"
]

def map_label(label):
    if label in VEHICLE_CLASSES:
        return "vehicle"
    return label