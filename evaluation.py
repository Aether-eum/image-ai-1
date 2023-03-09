
import os
import time

from PIL import Image
from ultralytics import YOLO

import torch
import numpy as np
from torchvision import transforms, utils
import skimage.transform
import skimage.color
import skimage
import torchvision
import torch.nn as nn
import csv



def main():
    thisdict = {}
    with open('imagesurveys.csv', newline='') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    print(len(your_list))
    print(your_list[0][3])
    print(your_list[0][6])

    # iterate through your_list
    for i in range(len(your_list)):
        # get the image file name
        img_file = your_list[i][3]
        # get the label value converted to int
        label_value = int(your_list[i][6])
        # check if the image file name is already in the dictionary
        if img_file in thisdict:
            thisdict[img_file].append(label_value)
        else:
            thisdict[img_file] = [label_value]

    root_dir = "/home/aether/pycharm/ai1"

    # load all the models we trained

    model_r50 = torchvision.models.resnet50(pretrained=True)
    model_r50.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))
    model_r50.load_state_dict(torch.load(os.path.join(root_dir, "resnet50_epoch_9.pth")))
    model_r50.eval()

    transform_model_r50 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
    ])

    model_r101 = torchvision.models.resnet101(pretrained=True)
    model_r101.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
    model_r101.load_state_dict(torch.load(os.path.join(root_dir, "resnet101_epoch_49.pth")))
    model_r101.eval()

    transform_r101 = transforms.Compose([  # [1]
        transforms.ToPILImage(),
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])


    model_yolon =  YOLO("runs/classify/yolov8n_v8_10e/weights/best.pt")  # load a custom model

    model_yolom =  YOLO("runs/classify/yolov8m_v8_20e/weights/best.pt")  # load a custom model

    model_yolol =  YOLO("runs/classify/yolov8l_v8_50e/weights/best.pt")  # load a custom model

    # read from the csv file and save the image file name and label value in a dictionary
    with open('imagetest.csv', newline='') as f:
        reader = csv.reader(f)
        test_list = list(reader)

    # create a csv file to save the results, overwrite if the file already exists
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Label", "Resnet50", "ConfidenceResnet50",
                         "Resnet101", "ConfidenceResnet101", "YOLOv8N", "ConfidenceYOLOv8N",
                         "YOLOv8M", "ConfidenceYOLOv8M", "YOLOv8L", "ConfidenceYOLOv8L"])

    accuracy_50 = 0
    accuracy_101 = 0
    accuracy_yolon = 0
    accuracy_yolom = 0
    accuracy_yolol = 0

    inference_time_50 = 0
    inference_time_101 = 0
    inference_time_yolon = 0
    inference_time_yolom = 0
    inference_time_yolol = 0

    mae_50 = 0
    mae_101 = 0
    mae_yolon = 0
    mae_yolom = 0
    mae_yolol = 0

    prep_time_50 = 0
    prep_time_101 = 0
    prep_time_yolon = 0
    prep_time_yolom = 0
    prep_time_yolol = 0


    # iterate through the test_list
    for i in range(len(test_list)):
        # get the image file name
        img_file = test_list[i][0]
        # get the expected label value converted to int
        label_value = int(test_list[i][1])
        path = os.path.join(root_dir, "images", img_file)
        img = Image.open(path)
        # convert to numpy array
        img = np.array(img)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        # transform the image
        start_time = time.time()
        img_50 = transform_model_r50(img)
        prep_time_50 += time.time() - start_time
        start_time = time.time()
        img_101 = transform_r101(img)
        prep_time_101 += time.time() - start_time


        # add batch dimension
        img_50 = img_50.unsqueeze(0)
        img_101 = img_101.unsqueeze(0)

        # pass the image through the models
        # measure the inference time
        start_time = time.time()
        output_50 = model_r50(img_50)
        inference_time_50 += time.time() - start_time
        start_time = time.time()
        output_101 = model_r101(img_101)
        inference_time_101 += time.time() - start_time

        # get the predicted class
        confidence_50, pred_50 = torch.max(output_50, 1)
        confidence_50 = torch.exp(confidence_50)

        confidence_101, pred_101 = torch.max(output_101, 1)
        confidence_101 = torch.exp(confidence_101)

        # get the predicted class
        pred_yolon = model_yolon(path)
        inference_time_yolon += pred_yolon[0].speed['inference']
        prep_time_yolon += pred_yolon[0].speed['preprocess']

        pred_yolom = model_yolom(path)
        inference_time_yolom += pred_yolom[0].speed['inference']
        prep_time_yolom += pred_yolom[0].speed['preprocess']
        pred_yolol = model_yolol(path)
        inference_time_yolol += pred_yolol[0].speed['inference']
        prep_time_yolol += pred_yolol[0].speed['preprocess']

        pred_yolon_arr = pred_yolon[0].probs.cpu().numpy()
        yolon_confidence = np.max(pred_yolon_arr)
        yolon_class = np.where(pred_yolon_arr == yolon_confidence)[0][0] + 1

        pred_yolom_arr = pred_yolom[0].probs.cpu().numpy()
        yolom_confidence = np.max(pred_yolom_arr)
        yolom_class = np.where(pred_yolom_arr == yolom_confidence)[0][0] + 1

        pred_yolol_arr = pred_yolol[0].probs.cpu().numpy()
        yolol_confidence = np.max(pred_yolol_arr)
        yolol_class = np.where(pred_yolol_arr == yolol_confidence)[0][0] + 1

        # print all
        print("Resnet50: ", pred_50.item(), confidence_50.item())
        print("Resnet101: ", pred_101.item(), confidence_101.item())
        print("YOLOn: ", yolon_class, yolon_confidence)
        print("YOLOm: ", yolom_class, yolom_confidence)
        print("YOLOl: ", yolol_class, yolol_confidence)

        # calculate the accuracy
        if pred_50.item() == label_value:
            accuracy_50 += 1
        if pred_101.item() == label_value:
            accuracy_101 += 1
        if yolon_class == label_value:
            accuracy_yolon += 1
        if yolom_class == label_value:
            accuracy_yolom += 1
        if yolol_class == label_value:
            accuracy_yolol += 1

        # calculate absolute difference
        mae_50 += abs(pred_50.item() - label_value)
        mae_101 += abs(pred_101.item() - label_value)
        mae_yolon += abs(yolon_class - label_value)
        mae_yolom += abs(yolom_class - label_value)
        mae_yolol += abs(yolol_class - label_value)

        # write to the csv file
        with open('results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([img_file, label_value, pred_50.item(), confidence_50.item(), pred_101.item(), confidence_101.item(),
                             yolon_class, yolon_confidence,yolom_class, yolom_confidence,yolol_class, yolol_confidence])


    # calculate the accuracy
    accuracy_50 = accuracy_50 / len(test_list)
    accuracy_101 = accuracy_101 / len(test_list)
    accuracy_yolon = accuracy_yolon / len(test_list)
    accuracy_yolom = accuracy_yolom / len(test_list)
    accuracy_yolol = accuracy_yolol / len(test_list)

    # calculate the inference time
    inference_time_50 = inference_time_50 / len(test_list)
    inference_time_101 = inference_time_101 / len(test_list)
    inference_time_yolon = inference_time_yolon / len(test_list)
    inference_time_yolom = inference_time_yolom / len(test_list)
    inference_time_yolol = inference_time_yolol / len(test_list)

    # calculate the prep time
    prep_time_yolon = prep_time_yolon / len(test_list)
    prep_time_yolom = prep_time_yolom / len(test_list)
    prep_time_yolol = prep_time_yolol / len(test_list)
    prep_time_50 = prep_time_50 / len(test_list)
    prep_time_101 = prep_time_101 / len(test_list)

    # convert to seconds
    inference_time_yolon = inference_time_yolon / 1000
    inference_time_yolom = inference_time_yolom / 1000
    inference_time_yolol = inference_time_yolol / 1000
    prep_time_yolon = prep_time_yolon / 1000
    prep_time_yolom = prep_time_yolom / 1000
    prep_time_yolol = prep_time_yolol / 1000


    # calculate the mean absolute error
    mae_50 = mae_50 / len(test_list)
    mae_101 = mae_101 / len(test_list)
    mae_yolon = mae_yolon / len(test_list)
    mae_yolom = mae_yolom / len(test_list)
    mae_yolol = mae_yolol / len(test_list)

    # write accuracy to the csv file
    with open('results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Accuracy", "-", accuracy_50, "-", accuracy_101,"-", accuracy_yolon,"-", accuracy_yolom,"-", accuracy_yolol,"-",])

    # write prep time to the csv file
    with open('results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Prep Time", "-", prep_time_50, "-", prep_time_101,"-", prep_time_yolon,"-", prep_time_yolom,"-", prep_time_yolol,"-",])

    # write inference time to the csv file
    with open('results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Inference Time", "-", inference_time_50, "-", inference_time_101,"-", inference_time_yolon,"-", inference_time_yolom,"-", inference_time_yolol,"-",])


    # write mean absolute error to the csv file
    with open('results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Mean Absolute Error", "-", mae_50, "-", mae_101,"-", mae_yolon,"-", mae_yolom,"-", mae_yolol,"-",])
































if __name__ == '__main__':
    main()