from ultralytics import YOLO

# Load the model.
# model = YOLO('yolov8n-cls.pt')
# model = YOLO('yolov8l-cls.pt')
model = YOLO('yolov8m-cls.pt')



# Training.
results = model.train(
    data='/home/aether/pycharm/ai1/dataset',
    task='classify',
    imgsz=448,
    epochs=20,
    batch=4,
    name='yolov8m_v8_20e'
)

# yolo task=classify mode=val model=runs/classify/yolov8l_v8_50e/weights/best.pt name=yolov8l_eval data=/home/aether/pycharm/ai1/dataset imgsz=1280
