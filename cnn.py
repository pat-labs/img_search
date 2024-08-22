# creating dataset
import os
import pandas as pd
import numpy as np

from sklearn import model_selection

train_images_path = "/dataset/flowers/train"

classes = os.listdir("/dataset/flowers/train")

image_ids = []
labels = []

for flower in classes:
    path = os.path.join(train_images_path, flower)
    temp = os.listdir(path)
    label = [flower] * len(temp)
    image_ids += temp
    labels += label

train_dataset = pd.DataFrame()
train_dataset["image_ids"] = image_ids
train_dataset["flowers"] = labels

label_map = {}
for i, flower in enumerate(classes):
    label_map[flower] = i

train_dataset["labels"] = train_dataset["flowers"].map(label_map)
train_dataset.to_csv("train_set.csv", index=False)
dfx = train_dataset.copy()


dfx["kfold"] = -1

dfx = dfx.sample(frac=1).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=dfx, y=dfx.labels.values)):
    print(len(trn_), len(val_))
dfx.loc[val_, "kfold"] = fold
