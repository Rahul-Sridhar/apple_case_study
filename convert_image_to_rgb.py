import os
import cv2
import numpy as np
import pandas as pd

root_dir = "/media/rahul/DATA/apple_case_study"
dataset_dir = os.path.join(root_dir, "squares/squares")
train_csv_filename = os.path.join(root_dir, "train_rgb.csv")
val_csv_filename = os.path.join(root_dir, "val_rgb.csv")
sub_dirs = ["train", "val"]
sub_sub_dirs = ["a", "b", "c"]
label_dict = {"a":1, "b":2, "c":3}

train_data = []
val_data = []

cnt = 0
for sub_dir in sub_dirs:
    for sub_sub_dir in sub_sub_dirs:
        folder_path = os.path.join(dataset_dir, sub_dir, sub_sub_dir)
        label = label_dict[sub_sub_dir]
        for filename in os.listdir(folder_path):
            cnt = cnt + 1
            print(cnt)
            image = cv2.imread(os.path.join(folder_path, filename))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Threshold the image - segment white background from post it notes
            _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV);
            # Find the contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                # If the contour is not really small, or really big
                h, w = image.shape[0], image.shape[1]
                # Get the four corners of the contour
                x, y, w, h = cv2.boundingRect(contour)
                roi = image[y:y + h, x:x + h]

                mean_blue = np.mean(roi[:, :, 0])
                mean_green = np.mean(roi[:, :, 1])
                mean_red = np.mean(roi[:, :, 2])

            if sub_dir == "train":
                train_data.append([mean_red, mean_green, mean_blue, label])
            else:
                val_data.append([mean_red, mean_green, mean_blue, label])

train_df = pd.DataFrame(train_data, columns=["Red", "Blue", "Green", "Class"])
val_df = pd.DataFrame(val_data, columns=["Red", "Blue", "Green", "Class"])

train_df.to_csv(train_csv_filename, encoding='utf-8', index=False)
val_df.to_csv(val_csv_filename, encoding='utf-8', index=False)


