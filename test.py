import os
from PIL import Image

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


    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(os.path.join(root_dir, "resnet50_epoch_9.pth")))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
    ])

    # load an image from images folder
    path = os.path.join(root_dir, "images", "1.png")
    img = Image.open(path)
    # convert to numpy array
    img = np.array(img)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    # transform the image
    img = transform(img)

    # add batch dimension
    img = img.unsqueeze(0)

    print(img.shape)
    # pass the image through the model
    output = model(img)

    # get the predicted class
    _, pred = torch.max(output, 1)
    #
    # # get the predicted class label
    # label = dataset.classes[pred.item()]

    print(output)

    print("Predicted class: ", pred)
    print("Trained labels: ", thisdict["1"])


if __name__ == '__main__':
    main()


