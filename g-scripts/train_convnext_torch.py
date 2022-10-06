import os
import datetime
import pytz
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from loguru import logger as lgr
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
lgr.info("No Warning Shown")

# Simple dataloader and label binarization, that is converting test labels into binary arrays of length 3 (number of classes) with 1 in places of applicable labels).
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
        return img, anno

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
    
# Here is an auxiliary function for checkpoint saving.
def checkpoint_save(model, save_path, epoch, f1):
    f = os.path.join(save_path, 'checkpoint-{:01d}-f1-{:02f}.pth'.format(epoch, f1))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    lgr.debug('saved checkpoint:', f)

# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }

def evaluate_model(model, val_dataloader):
    model.eval()
    with torch.no_grad():
        model_result = []
        targets = []
        for imgs, batch_targets in val_dataloader:
            imgs = imgs.to(device)
            model_batch_result = model(imgs)
            model_result.extend(model_batch_result.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())

    result = calculate_metrics(np.array(model_result), np.array(targets))
    for metric in result:
        logger.add_scalar('val/' + metric, result[metric], iteration)
    lgr.success("epoch:{:2d} iter:{:3d} val/: micro f1: {:.3f} macro f1: {:.3f} samples f1: {:.3f}".format(epoch, iteration, result['micro/f1'], result['macro/f1'], result['samples/f1']))
    return result['samples/f1']


run_id = 'retrain_convnext_large' 
data_dir = f"./training"
image_dir = f'{data_dir}/images/'
annot_dir = f'{data_dir}/csvs'
result_dir = f'./results/{run_id}'
train_annotations = os.path.join(annot_dir, 'train_masked/train_masked.csv')


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

device = torch.device('cuda')
# Save path for checkpoints
save_path = f'{result_dir}/checkpoints/'
# Save path for logs
logdir = f'{result_dir}/logs/'

Path(save_path).mkdir(parents=True, exist_ok=True)
Path(logdir).mkdir(parents=True, exist_ok=True)

# Run tensorboard
#%load_ext tensorboard
#%tensorboard --logdir {logdir}

# Data augmentation and normalization for training
# Just normalization for validation
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


# Initialize the dataloaders for training.


full_df = pd.read_csv(train_annotations)
train_df = full_df.sample(frac=0.8, random_state=42)
valid_df = full_df[~full_df.apply(tuple,1).isin(train_df.apply(tuple,1))]

train_path = "./train.csv"
valid_path = "./valid.csv"

train_df.to_csv(train_path, index=False)
valid_df.to_csv(valid_path, index=False)


val_dataset = RoofConditionDataset(f'{image_dir}/train_masked', valid_path, data_transforms['val'])

train_dataset = RoofConditionDataset(f'{image_dir}/train_masked', train_path, data_transforms['train'])


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

num_train_batches = int(np.ceil(len(train_dataset) / batch_size))

test_freq = num_train_batches # Test model frequency (iterations)
progress_freq = num_train_batches/2

# Initialize the model
model = ConvnxtLarge(len(train_dataset.classes))
# Switch model to the training mode and move it to GPU.
model.train()
model = model.to(device)

n_iteration = num_train_batches * max_epoch_number
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iteration, eta_min=0, last_epoch=- 1, verbose=False)

# If more than one GPU is available we can use both to speed up the training.
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


# Loss function
criterion = nn.BCELoss()
# Tensoboard logger
logger = SummaryWriter(logdir)


# Run training
epoch = 0
iteration = 0
best_f1 = 0
while True:
    batch_losses = []
    epoch_f1 = []
    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        model_result = model(imgs)
        loss = criterion(model_result, targets.type(torch.float))

        batch_loss_value = loss.item()
        loss.backward()
        optimizer.step()

        logger.add_scalar('train_loss', batch_loss_value, iteration)
        batch_losses.append(batch_loss_value)
        with torch.no_grad():
            result = calculate_metrics(model_result.cpu().numpy(), targets.cpu().numpy())
            for metric in result:
                logger.add_scalar('train/' + metric, result[metric], iteration)

        if iteration % progress_freq == 0:
            lgr.debug(f"{datetime.datetime.now(pytz.timezone('Asia/Kolkata'))} epoch:{epoch:2d} iter:{iteration:3d} train: loss:{batch_losses[-1]:.3f}")
        
        iteration += 1

    loss_value = np.mean(batch_losses)

    if epoch_f1:
        epoch_f1_value = epoch_f1[-1]
    else:
        epoch_f1_value = evaluate_model(model, val_dataloader)
        model.train()

    lgr.success("{} / epoch:{:2d} iter:{:3d} train: loss:{:.3f} val_f1: {:.3f}".format(datetime.datetime.now(pytz.timezone('Asia/Kolkata')), epoch, iteration, loss_value, epoch_f1_value))
    if (epoch % save_freq == 0) or (epoch_f1_value > best_f1):
        checkpoint_save(model, save_path, epoch, epoch_f1_value)
    if epoch_f1_value > best_f1:
        best_f1 = epoch_f1_value

    scheduler.step()
    
    epoch += 1
    if max_epoch_number <= epoch:
        break