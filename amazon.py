import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils import data
from torchvision.datasets import cifar
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

print(f'is CUDA available? {torch.cuda.is_available()}')
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

root = Path('/data/dataset/amazon')
train = root / 'train'
test = root / 'test'
bs = 10
train_df = pd.read_csv(root / 'train_v2.csv')
labels = pd.Series(train_df['tags'].values, index=train_df['image_name'])
labels = {k: v.split() for k, v in labels.iteritems()}
tags = list(labels.values())
tags = tuple(set(item for sublist in tags for item in sublist))
tags2index = {tag: i for i, tag in enumerate(tags)}
print(f'tag lens = {len(tags)}')


def encode_tags(input_tags):
    out = torch.empty(1, len(tags), dtype=torch.float)
    for t in input_tags:
        out[:, tags2index[t]] = 1
    out = out.squeeze(0)
    return out


fns = list(labels.keys())
print(f'len of dataset = {len(fns)}')
idxs = list(range(len(fns)))
idxs2fn = {i: fn for i, fn in zip(idxs, fns)}
validation_split = .2
random_seed = 42
split = int(np.floor(validation_split * len(fns)))
np.random.seed(random_seed)
np.random.shuffle(idxs)
train_ids, val_ids = idxs[split:], idxs[:split]
train_sampler = SubsetRandomSampler(train_ids)
val_sampler = SubsetRandomSampler(val_ids)


class AmazonDataset(data.Dataset):
    def __init__(self, transform):
        self.transform = transform

    def __len__(self):
        return len(labels)

    def __getitem__(self, index):
        fn = fns[index]
        fn_tags = labels[fn]
        fn_tags = encode_tags(fn_tags)
        img = Image.open(train / f'{fn}.jpg')

        if self.transform:
            img = self.transform(img)
        img = img[:3, ...]
        return img, fn_tags


trainset = AmazonDataset(transform=tfms)
train_loader = data.DataLoader(trainset, batch_size=bs, sampler=train_sampler)
val_loader = data.DataLoader(trainset, batch_size=bs, sampler=val_sampler)
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=len(tags)),
    nn.Sigmoid()
)
model.cuda()
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
crit = nn.L1Loss()
epocs = 1
val_loss = []
print_debug = 10
for e in range(epocs):
    train_loss = []
    for i, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        yh = model(x)
        loss = crit(yh, y)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % print_debug == 0:
            print(f'[{e:5d}] train = {np.mean(train_loss):.5f}')
