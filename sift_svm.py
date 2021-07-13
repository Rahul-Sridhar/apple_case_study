import os
import cv2
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

root_dir = "/media/rahul/DATA/apple_case_study"
dataset_dir = os.path.join(root_dir, "squares/squares")
sub_dirs = ["train", "val"]
sub_sub_dirs = ["a", "b", "c"]
label_dict = {"a":1, "b":2, "c":3}


def CalcFeatures(img, th):
    sift = cv2.xfeatures2d.SIFT_create(th)
    kp, des = sift.detectAndCompute(img, None)
    return des

def bag_of_features(features, centres, k=500):
    vec = np.zeros((1, k))
    for i in range(features.shape[0]):
        feat = features[i]
        diff = np.tile(feat, (k, 1)) - centres
        dist = pow(((pow(diff, 2)).sum(axis=1)), 0.5)
        idx_dist = dist.argsort()
        idx = idx_dist[0]
        vec[0][idx] += 1
    return vec

def sift_svm(thresh):
    t0 = time.time()
    train_features = []
    val_features = []
    cnt = 0
    for sub_dir in sub_dirs:
        for sub_sub_dir in sub_sub_dirs:
            folder_path = os.path.join(dataset_dir, sub_dir, sub_sub_dir)
            for filename in os.listdir(folder_path):
                cnt = cnt + 1
                # print(cnt)
                image = cv2.imread(os.path.join(folder_path, filename), 0)
                img_des = CalcFeatures(image, thresh)
                if img_des is not None:
                    if sub_dir == "train":
                        train_features.append(img_des)
                    else:
                        val_features.append(img_des)

    train_features = np.vstack(train_features)
    val_features = np.vstack(val_features)

    k = 150
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    train_compactness, train_labels, train_centres = cv2.kmeans(train_features, k, None, criteria, 10, flags)
    val_compactness, val_labels, val_centres = cv2.kmeans(val_features, k, None, criteria, 10, flags)

    X_train = []
    X_val = []
    y_train = []
    y_val = []

    for sub_dir in sub_dirs:
        for sub_sub_dir in sub_sub_dirs:
            folder_path = os.path.join(dataset_dir, sub_dir, sub_sub_dir)
            label = label_dict[sub_sub_dir]
            for filename in os.listdir(folder_path):
                cnt = cnt + 1
                # print(cnt)
                image = cv2.imread(os.path.join(folder_path, filename), 0)
                img_des = CalcFeatures(image, thresh)
                if sub_dir == "train":
                    if img_des is not None:
                        img_vec = bag_of_features(img_des, train_centres, k)
                        X_train.append(img_vec)
                        y_train.append(label)
                else:
                    if img_des is not None:
                        img_vec = bag_of_features(img_des, val_centres, k)
                        X_val.append(img_vec)
                        y_val.append(label)

    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)

    clf = SVC()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    conf_mat = confusion_matrix(y_val, preds)

    t1 = time.time()

    return acc * 100, conf_mat, (t1 - t0)

def main():
    accuracy = []
    timer = []
    for i in range(5, 26, 5):
        print('\nCalculating for a threshold of {}'.format(i))
        data = sift_svm(i)
        accuracy.append(data[0])
        conf_mat = data[1]
        timer.append(data[2])
        print('\nAccuracy = {}\nTime taken = {} sec\nConfusion matrix :\n{}'.format(data[0], data[2], data[1]))

if __name__ == "__main__":
    main()