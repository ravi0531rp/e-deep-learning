from glob import glob
import os
import torch
import torch.nn as nn
import torchvision
import datetime
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import max_error, precision_score, recall_score, f1_score

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
thresh = 0.5
PATH = './results/'
path = PATH.split("/")
model_name  = path[2] + "-" + path[4].replace(".pth","").replace(".","_")

class RoofConditionDataset(Dataset):
    def __init__(self, data_path, anno_path, transforms):
        self.transforms = transforms
        annot_df = pd.read_csv(anno_path)

        self.classes = [
            "CLASS_A",
            "CLASS_B",
            "CLASS_C"
        ]

        self.imgs = []
        self.annos = []
        self.data_path = data_path

        order = ["CLASS_A",
            "CLASS_B",
            "CLASS_C"]

        orig_labels = annot_df["label"].to_list()
        pred_y = []
        for pr in orig_labels:
            pr = pr.replace("'","").replace("[","").replace("]","").split(",")
            tmp_pr = [0]*len(order)
            if len(pr):
                for idx,elem in enumerate(order):
                    if elem in pr:
                        tmp_pr[idx] = 1
    
            pred_y.append(tmp_pr)
        
        annot_df["bin_labels"] = pred_y

        for idx, row in annot_df.iterrows():
            self.imgs.append(row['image'])
            self.annos.append(np.array(row["bin_labels"], dtype=float))


    def __getitem__(self, item):
        anno = self.annos[item]
        img_path = os.path.join(self.data_path, self.imgs[item])
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, anno, self.imgs[item] # added name for debugging

    def __len__(self):
        return len(self.imgs)

# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a Sigmoid instead of a default Softmax.
class ConvnxtLarge(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        convnext = models.convnext_large(pretrained=True)
        convnext.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(convnext.classifier[2].in_features, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, out_features=n_classes)
        )
        self.base_model = convnext
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))




# Initialize the training parameters.
num_workers = 8 # Number of CPU processes for data preprocessing
lr = 5e-5 # Learning rate
batch_size = 16
save_freq = 5 # Save checkpoint frequency (epochs)
#test_freq = 200 # Test model frequency (iterations)
max_epoch_number = 51 # Number of epochs for training 
# Note: on the small subset of data overfitting happens after 30-35 epochs

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
model = ConvnxtLarge(9)


model.load_state_dict(torch.load(PATH))
model.eval()
logger.success("Model Created and weights loaded..")


if torch.cuda.is_available():
    model.cuda()
    logger.info("Eval using GPU")
else:
    logger.warning("Eval using CPU")



test_csv = "./bench_masked/masked.csv"
test_annotations = pd.read_csv(test_csv) 

test_dataset = RoofConditionDataset(f'./bench_masked', test_csv, data_transforms['test'])

test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers)



genres = [
            "CLASS_A",
            "CLASS_B",
            "CLASS_C"
        ]

logger.info("Generating Predictions..")

values = {}
# for counter, data in enumerate(test_loader):
for image, target, name in test_dataloader:
    # image, target, names = data['image'].to(device), data['label'].to(device), data['names']
    image, target = image.to(device), target.to(device)
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    start_time = datetime.datetime.now()
    # get the predictions by passing the image through the model
    outputs = model(image)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000
    # outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()

    out_thr = [1 if i>=thresh else 0 for i in outputs[0]]
    # logger.debug(out_thr)
    values[name[0]] = out_thr
    if len(values) % 200 == 0:
        logger.info(f"Done for {len(values)}")

column_values = [
            "CLASS_A",
            "CLASS_B",
            "CLASS_C"
        ]

def lst2str(cols, lst):
    new_lst = []
    for idx in range(len(lst)):
        if lst[idx] == 1:
            new_lst.append("'"+str(cols[idx])+"'")
    return '"[' + ",".join(new_lst) + ']"'


combined = []

for k,v in values.items():
    combined.append(k+","+ lst2str(column_values, v))

logger.debug(combined[0])

with open(f'./bench_{model_name}_{str(thresh).replace(".","_")}.csv','w') as fw:
    fw.write("image,Preds\n")
    for val in combined:
        fw.write(val+"\n")