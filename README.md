# 🌦️ Aerial Object Detection under Adverse Weather Conditions

## 📌 Overview

This project focuses on **aerial object detection in challenging weather conditions** such as fog, rain, and motion blur. Detecting objects from aerial images becomes difficult when visibility is reduced, making it a critical problem for applications like surveillance, traffic monitoring, and disaster management.

To address this, the project compares two popular deep learning models — **YOLOv8** and **Faster R-CNN** — and evaluates their performance under different environmental conditions. The study highlights the trade-off between **speed and robustness**, where YOLOv8 is faster and suitable for real-time detection, while Faster R-CNN performs more reliably in complex scenarios.

---

## 🎯 Objectives

* Perform object detection on aerial images under adverse weather conditions
* Compare YOLOv8 and Faster R-CNN
* Analyze performance using metrics like Precision, Recall, and mAP
* Study the impact of weather conditions on detection accuracy

---

## 🧠 Models Used

* **YOLOv8** – Fast, single-stage detector for real-time applications
* **Faster R-CNN** – Accurate, two-stage detector with better robustness

---

## 📂 Project Structure

```
aerial_weather_comparative_detection/
│── frontend/
│── inference/
│── utils/
│── .streamlit/
│── requirements.txt
│── README.md
```

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/Nandini-Alapati/aerial_weather_comparative_detection.git

# Navigate to project folder
cd aerial_weather_comparative_detection

# Install dependencies
pip install -r requirements.txt

# Run the application
python frontend/app.py
```

---

## 📊 Results

* YOLOv8 performs better in terms of **speed and real-time detection**
* Faster R-CNN shows **better robustness under fog, rain, and blur**
* The comparison demonstrates a clear trade-off between **efficiency and accuracy**

---

## ⚠️ Note

Due to GitHub file size limitations, **model weights and dataset are not included** in this repository.

👉 You can download them from:
(Add your Google Drive link here)

---

## 🛠️ Technologies Used

* Python
* PyTorch
* OpenCV
* YOLOv8 (Ultralytics)
* Faster R-CNN (Torchvision)

---

## 👩‍💻 Author

**Nandini Alapati**
B.Tech CSE (AI & ML)

**Sharon Bezawada**
B.Tech CSE (AI & ML)

**Kandula Chandra Sekhar**
B.Tech CSE (AI & ML)

---

## ⭐ Conclusion

This project demonstrates how different object detection models perform under adverse weather conditions. It provides useful insights for selecting appropriate models in real-world aerial surveillance systems, balancing between **speed and reliability**.

---
