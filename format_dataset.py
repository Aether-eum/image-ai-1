import csv
import os
import random


def generate_data(dataset_dir):
    # create test, train and valid directories if they don't exist
    if not os.path.exists(os.path.join(dataset_dir, 'train')):
        os.makedirs(os.path.join(dataset_dir, 'train'))
    if not os.path.exists(os.path.join(dataset_dir, 'test')):
        os.makedirs(os.path.join(dataset_dir, 'test'))
    if not os.path.exists(os.path.join(dataset_dir, 'valid')):
        os.makedirs(os.path.join(dataset_dir, 'valid'))

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

    # separate the keys and values of the dictionary into train, test and valid dictionaries randomly
    train_dict = {}
    test_dict = {}
    valid_dict = {}
    for key, value in thisdict.items():
        rand = random.random()
        if rand < 0.7:
            train_dict[key] = value
        elif rand < 0.85:
            test_dict[key] = value
        else:
            valid_dict[key] = value

    train_path = os.path.join(dataset_dir, "train")
    images_path = "/home/aether/pycharm/ai1/images"

    for key, value in train_dict.items():
        # create the directory if not exists using the label average value as name
        label = int(sum(value) / len(value))
        label_path = os.path.join(train_path, str(label))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        # copy the image file to the label directory
        os.system("cp " + os.path.join(images_path, "{}.png".format(key)) + " " + label_path)

    test_path = os.path.join(dataset_dir, "test")
    for key, value in test_dict.items():
        # create the directory if not exists using the label average value as name
        label = int(sum(value) / len(value))
        label_path = os.path.join(test_path, str(label))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        # copy the image file to the label directory
        os.system("cp " + os.path.join(images_path, "{}.png".format(key)) + " " + label_path)

    valid_path = os.path.join(dataset_dir, "valid")
    for key, value in valid_dict.items():
        # create the directory if not exists using the label average value as name
        label = int(sum(value) / len(value))
        label_path = os.path.join(valid_path, str(label))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        # copy the image file to the label directory
        os.system("cp " + os.path.join(images_path, "{}.png".format(key)) + " " + label_path)





if __name__ == '__main__':
    generate_data("/home/aether/pycharm/ai1/dataset")