import json

from ultralytics import YOLO
import numpy as np
import os


# assumes a directory called assets that contains the images and the jsons
def main():
    root_dir = "/home/aether/pycharm/ai1"
    if not os.path.exists(os.path.join(root_dir, "combined")):
        os.mkdir(os.path.join(root_dir, "combined"))
    if not os.path.exists(os.path.join(root_dir, "combined", "images")):
        os.mkdir(os.path.join(root_dir, "combined", "images"))
    if not os.path.exists(os.path.join(root_dir, "combined", "json")):
        os.mkdir(os.path.join(root_dir, "combined", "json"))


    image_dir = "/home/aether/pycharm/ai1/assets/images"
    json_dir = "/home/aether/pycharm/ai1/assets/json"

    image_extra_dir = "/home/aether/pycharm/ai1/build/images"
    json_extra_dir = "/home/aether/pycharm/ai1/build/json"

    meta_dir = "/home/aether/pycharm/ai1/assets/json/_metadata.json"
    meta_extra_dir = "/home/aether/pycharm/ai1/build/json/_metadata.json"
    combined_meta_dir = "/home/aether/pycharm/ai1/combined/json/_metadata.json"

    # load json array as list
    with open(meta_dir) as f:
        data = json.load(f)

    with open(meta_extra_dir) as f:
        data_extra = json.load(f)

    print(len(data))
    print(len(data_extra))

    # move images from build to assets
    for root, dirs, files in os.walk(image_extra_dir):
        # iterate through png files
        for file in files:
            if file.endswith(".png"):
                os.rename(os.path.join(root, file), os.path.join(image_dir, file))

    # move json from build to assets
    for root, dirs, files in os.walk(json_extra_dir):
        # iterate through png files
        for file in files:
            if file.endswith(".json"):
                # if file is _metadata.json rename it to _metadata plus the length of the json array
                if file == "_metadata.json":
                    os.rename(os.path.join(root, file), os.path.join(json_dir, "_metadata" + str(len(data)) + ".json"))
                else:
                    os.rename(os.path.join(root, file), os.path.join(json_dir, file))

    # combine json arrays
    data = data + data_extra
    print(len(data))

    # shuffle the combined json array
    np.random.shuffle(data)
    print(len(data))
    print(data[0])

    # for each image in the combined json array move the image to combined/images and rename it to the index of the image in the array. do the same for the json file
    for i in range(len(data)):
        os.rename(os.path.join(image_dir, str(data[i]["edition"]) + ".png"), os.path.join(root_dir, "combined", "images", str(i) + ".png"))
        os.rename(os.path.join(json_dir, str(data[i]["edition"]) + ".json"), os.path.join(root_dir, "combined", "json", str(i) + ".json"))
        data[i]["name"] = "Vibe Kingdom #" + str(i)
        data[i]["edition"] = i
        data[i]["image"] = "ipfs://NewUriToReplace/" +  str(i) + ".png"
        # edit the json file to have the new name and edition number
        with open(os.path.join(root_dir, "combined", "json", str(i) + ".json"), 'w') as f:
            json.dump(data[i], f)

    # save list as json array
    with open(combined_meta_dir, 'w') as f:
        json.dump(data, f)

    # # delete the build folder
    # os.rmdir(os.path.join(root_dir, "build", "images"))
    # os.rmdir(os.path.join(root_dir, "build", "json"))
    # os.rmdir(os.path.join(root_dir, "build"))
    #
    # # delete the assets folder
    # os.rmdir(os.path.join(root_dir, "assets", "images"))
    # os.rmdir(os.path.join(root_dir, "assets", "json"))
    # os.rmdir(os.path.join(root_dir, "assets"))
    #
    # # rename the combined folder to assets
    # os.rename(os.path.join(root_dir, "combined", "images"), os.path.join(root_dir, "assets", "images"))
    # os.rename(os.path.join(root_dir, "combined", "json"), os.path.join(root_dir, "assets", "json"))
    # os.rename(os.path.join(root_dir, "combined"), os.path.join(root_dir, "assets"))


if __name__ == '__main__':
    main()

