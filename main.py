import os
import csv
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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
            # get the label value
            label_value = your_list[i][6]
            # append the image file name to self.images if image was not already added
            img_name = os.path.join(root_dir, "{}.png".format(img_file))
            self.images.append(img_name)
            # append the label value to self.labels
            self.labels.append(label_value)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        print(img)
        print(label)

        if self.transform:
            img = self.transform(img)

        return img, label




def train_model():


    # create a dataset object
    dataset = ImageDataset(root_dir="./images")
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
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
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


