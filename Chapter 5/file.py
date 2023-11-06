from ultralytics import YOLO
import cv2

model = YOLO("../YOLO/yolov8l.pt")
results = model("../images/3.png", show=True)
cv2.waitKey(0)