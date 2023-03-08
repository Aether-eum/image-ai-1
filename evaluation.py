
import os
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

    model_eb7 = torchvision.models.efficientnet_b7(pretrained=True)
    model_eb7.fc = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(512, 10),
                                  nn.LogSoftmax(dim=1))
    model_eb7.load_state_dict(torch.load(os.path.join(root_dir, "efficientnetb7_epoch_9.pth")))
    model_eb7.eval()

    transform_eb7 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(896),
        transforms.ToTensor(),
    ])

    model_yolon =  YOLO("runs/classify/yolov8n_v8_10e/weights/best.pt")  # load a custom model

    model_yolom =  YOLO("runs/classify/yolov8m_v8_20e/weights/best.pt")  # load a custom model

    model_yolol =  YOLO("runs/classify/yolov8l_v8_50e/weights/best.pt")  # load a custom model

    path = os.path.join(root_dir, "images", "642.png")
    img = Image.open(path)
    # convert to numpy array
    img = np.array(img)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    # transform the image
    img_50 = transform_model_r50(img)
    img_101 = transform_r101(img)
    # img_eb7 = transform_eb7(img)

    # add batch dimension
    img_50 = img_50.unsqueeze(0)
    img_101 = img_101.unsqueeze(0)
    # img_eb7 = img_eb7.unsqueeze(0)

    # pass the image through the models
    output_50 = model_r50(img_50)
    output_101 = model_r101(img_101)
    # output_eb7 = model_eb7(img_eb7)

    # get the predicted class
    _, pred_50 = torch.max(output_50, 1)
    _, pred_101 = torch.max(output_101, 1)
    # _, pred_eb7 = torch.max(output_eb7, 1)

    # # get the predicted class
    pred_yolon = model_yolon(path)
    # print(pred_yolon)
    # _, pred_yolon = torch.max(pred_yolon., 1)
    # print(pred_yolon)
    pred_yolom = model_yolom(path)
    pred_yolol = model_yolol(path)

    # # get the predicted class
    # _, pred_yolon = torch.max(pred_yolon, 1)
    # _, pred_yolom = torch.max(pred_yolom, 1)
    # _, pred_yolol = torch.max(pred_yolol, 1)
    #
    # # get the predicted class
    # pred_yolon = pred_yolon.item()
    # pred_yolom = pred_yolom.item()
    # pred_yolol = pred_yolol.item()

    # print all
    print("Resnet50: ", pred_50.item())
    print("Resnet101: ", pred_101.item())
    # print("Efficientnetb7: ", pred_eb7.item())
    # print("YOLOn: ", pred_yolon)
    # print("YOLOm: ", pred_yolom)
    # print("YOLOl: ", pred_yolol)



















if __name__ == '__main__':
    main()