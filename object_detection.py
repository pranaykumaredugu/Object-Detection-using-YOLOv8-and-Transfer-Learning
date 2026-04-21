import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.animation as animation
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
fig, ax = plt.subplots()
def update_frame(i):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    
    results = model(frame, imgsz=640)

    boxes = results[0].boxes.xyxy.cpu().numpy()  
    scores = results[0].boxes.conf.cpu().numpy() 
    classes = results[0].boxes.cls.cpu().numpy()  

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        label = f"{model.names[int(cls)]} {score:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    ax.clear()
    
   
    ax.imshow(frame_rgb)
    ax.axis('off')


ani = animation.FuncAnimation(fig, update_frame, interval=50)

plt.show()

cap.release()
