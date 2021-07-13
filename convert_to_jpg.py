import os
import cv2

root_dir = "/media/rahul/DATA/apple_case_study/squares/squares"
sub_dirs = ["train", "val"]
sub_sub_dirs = ["a_old", "b_old", "c_old"]

cnt = 0
for sub_dir in sub_dirs:
    for sub_sub_dir in sub_sub_dirs:
        folder_path = os.path.join(root_dir, sub_dir, sub_sub_dir)
        new_sub_sub_dir = sub_sub_dir.replace("_old", "")
        new_folder_path = os.path.join(root_dir, sub_dir, new_sub_sub_dir)
        for filename in os.listdir(folder_path):
            cnt = cnt + 1
            print(cnt)
            image = cv2.imread(os.path.join(folder_path, filename))
            filename = filename.replace(".png", ".jpg")
            filename = filename.replace(".bmp", ".jpg")
            cv2.imwrite(os.path.join(new_folder_path, filename), image)