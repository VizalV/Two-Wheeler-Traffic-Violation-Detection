import ultralytics
ultralytics.checks()
from ultralytics import YOLO
model=YOLO('yolov8s.yaml')
results = model.train(
    data='/home/vishal.v/work/TW_Detection/config.yaml',
    imgsz=640,
    epochs=75,
    device='0,1',
    save=True
)
