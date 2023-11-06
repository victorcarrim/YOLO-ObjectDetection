from ultralytics import YOLO
import cv2
import cvzone
import math

cv2.namedWindow("preview")
# cap = cv2.VideoCapture(0) # Webcam
cap = cv2.VideoCapture("../Videos/people.mp4") # Video
# cap.set(3, 1280)
# cap.set(4, 720)

model = YOLO("../YOLO/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Falha ao capturar o frame da câmera.")
            break

        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:

                #Bounding Box
                ## Feito com o cv2
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                ## Feito com o cvzone do video
                w, h = x2-x1, y2-y1
                cvzone.cornerRect(frame, (x1, y1, w, h))

                #Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                #Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(40 , y1)), scale=1.5, thickness=2)


        cv2.imshow("preview", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrompido pelo usuário")

# Quando tudo estiver feito, libera o vídeo capturado e destrói as janelas
cap.release()
cv2.destroyAllWindows()
