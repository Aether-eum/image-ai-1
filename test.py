from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO("runs/classify/yolov8n_v8_10e/weights/best.pt")  # load a custom model

# Predict with the model
results = model("/home/aether/pycharm/ai1/dataset/valid/5/191.png")  # predict on an image

print(results)

# get pred tensor
pred = results[0].probs
# from tensor to numpy
pred_arr = pred.cpu().numpy()
# get the max value
print(pred_arr)
pred = np.max(pred_arr)
print(pred)
# find the index of the max value
pred_index = np.where(pred_arr == pred)
# print the index
print(pred_index[0][0]+1, pred)