from ultralytics import YOLO

# Load a model
model = YOLO("runs/classify/yolov8m_v8_20e/weights/best.pt")  # load a custom model

# Predict with the model
results = model("/home/aether/pycharm/ai1/dataset/valid/5/191.png")  # predict on an image

print(results)