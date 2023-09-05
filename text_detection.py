from ultralytics import YOLO
from utils.pre_processing import get_input_data

model = YOLO("yolo_pretrained/yolov8n.pt")
input_img = get_input_data()

for i in range(len(input_img)):
    results = model(input_img[i])
    print(results)