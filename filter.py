import json

from ultralytics import YOLO
import numpy as np
import os


# assumes a directory called assets that contains the images and the jsons
def main():
    root_dir = "/home/aether/pycharm/ai1"

    if not os.path.exists(os.path.join(root_dir, "excluded")):
        os.mkdir(os.path.join(root_dir, "excluded"))
    if not os.path.exists(os.path.join(root_dir, "excluded", "images")):
        os.mkdir(os.path.join(root_dir, "excluded", "images"))
    if not os.path.exists(os.path.join(root_dir, "excluded", "json")):
        os.mkdir(os.path.join(root_dir, "excluded", "json"))


    # Load model
    model = YOLO("runs/classify/yolov8l_v8_50e/weights/best.pt")
    # iterate through the images
    image_dir = "/home/aether/pycharm/ai1/assets/images"
    json_dir = "/home/aether/pycharm/ai1/assets/json"
    meta_dir = "/home/aether/pycharm/ai1/assets/json/_metadata.json"

    # load json array as list
    with open(meta_dir) as f:
        data = json.load(f)

    print(len(data))

    valid_count = 0
    invalid_count = 0

    for root, dirs, files in os.walk(image_dir):
        # iterate through png files
        for file in files:
            if file.endswith(".png"):
                # predict the image
                results = model(os.path.join(root, file))
                # get pred tensor
                pred = results[0].probs
                # from tensor to numpy
                pred_arr = pred.cpu().numpy()
                # get the max value
                pred = np.max(pred_arr)
                # find the index of the max value
                pred_index = np.where(pred_arr == pred)
                if pred_index[0][0] + 1 < 4:
                    # move the image to excluded folder
                    list_name = 'Vibe Kingdom #' + str(int(file[:-4]))
                    print(list_name)
                    # find the index of the list where the name equals list_name
                    list_index = next((index for (index, d) in enumerate(data) if d["name"] == list_name), None)
                    print(list_index)
                    print(data[list_index])
                    data.pop(list_index)
                    os.rename(os.path.join(root, file), os.path.join(root_dir, "excluded", "images", file))
                    # move the json to excluded folder
                    os.rename(os.path.join(json_dir, file[:-4] + ".json"), os.path.join(root_dir, "excluded", "json", file[:-4] + ".json"))
                    print("Moved: ", file, pred_index[0][0] + 1)
                    invalid_count += 1
                else:
                    valid_count += 1

    print("Valid count: ", valid_count)
    print("Invalid count: ", invalid_count)

    # save list as json array
    with open(meta_dir, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()

