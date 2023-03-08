
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

    model_yolo = YOLO("runs/classify/yolov8m_v8_20e/weights/best.pt")













if __name__ == '__main__':
    main()