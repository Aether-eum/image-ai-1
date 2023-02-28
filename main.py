import os
import csv

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import skimage.io as io
import skimage.transform
import skimage.color
import skimage
from torchvision import transforms, utils


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        thisdict = {}
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

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

        # iterate through the dictionary
        for key, value in thisdict.items():
            # print(key, value)
            # append the image file name to self.images if image was not already added
            img_name = "{}.png".format(key)
            self.images.append(img_name)
            # append the int average label value to self.labels
            self.labels.append(int(np.mean(value)))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def load_image(self, image_index):
        # load image using skimage
        path = os.path.join(self.root_dir, "images",  self.images[image_index])
        img = io.imread(path)

        # remove alpha channel
        if img.shape[2] == 4:
            img = img[:, :, :3]

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img


def train_model():


    # create a dataset object
    transform = transforms.ToTensor()

    dataset = ImageDataset(root_dir="/home/aether/pycharm/ai1", transform=transform)
    # split the dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    # create pytorch model
    model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # Train the model
    for epoch in range(10):
        print("Epoch: ", epoch)
        model.train()
        for i, (image, label) in enumerate(train_dataloader):
            # print(images)
            # print(labels)
            print(i, image.size(), label.size())

            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Step: ", i, "Loss: ", loss.item())

        # Validate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print("Accuracy: ", correct / total)

    # Save the model with epoch number
    torch.save(model.state_dict(), "model_epoch_{}.pth".format(epoch))


if __name__ == '__main__':
    train_model()


