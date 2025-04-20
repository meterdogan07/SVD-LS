from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import math
import random
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, Subset
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet101, ResNet101_Weights
from torchvision.models import vit_l_16, ViT_L_16_Weights, vit_b_16, ViT_B_16_Weights

from collections import namedtuple, Counter
import torch.nn.functional as F
from torch.nn.functional import relu
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from sklearn.model_selection import KFold

from IPython.core.debugger import Pdb
import sys
import cv2 as cv
import pickle
import collections
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 448
torch.manual_seed(random_seed)

with open("train_data.pickle", 'rb') as file:
    loaded_data1 = pickle.load(file)
    train_dataset = loaded_data1
    
with open("test_data.pickle", 'rb') as file:
    loaded_data2 = pickle.load(file)
    test_dataset = loaded_data2


train_dataloader2 = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Perform Data Augmentation
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
])

def get_class_distribution(dataset):
    labels = [label.item() for _, label in dataset]
    return Counter(labels)

def augment_class(dataset, class_indices, augmentations, num_augmented_samples):
    augmented_images = []
    augmented_labels = []
    subset = Subset(dataset, class_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    
    for images, labels in loader:
        for _ in range(num_augmented_samples // len(loader) + 1):
            augmented_batch = augmentations(images)
            augmented_images.append(augmented_batch)
            augmented_labels.append(labels)

    augmented_images = torch.cat(augmented_images)[:num_augmented_samples]
    augmented_labels = torch.cat(augmented_labels)[:num_augmented_samples]
    return augmented_images, augmented_labels

def augment_dataset(train_dataset):
    class_distribution = get_class_distribution(train_dataset)
    print("Original class distribution:", class_distribution)
    logger.info(f"Original class distribution: {class_distribution}")

    target_num_samples = max(class_distribution.values())

    augmented_images = []
    augmented_labels = []
    for class_label, count in class_distribution.items():
        if count < target_num_samples:
            num_augmented_samples = target_num_samples - count
            class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_label]
            images, labels = augment_class(train_dataset, class_indices, augmentations, num_augmented_samples)
            augmented_images.append(images)
            augmented_labels.append(labels)

    # Convert augmented data to tensors
    if augmented_images:
        augmented_images_tensor = torch.cat(augmented_images)
        augmented_labels_tensor = torch.cat(augmented_labels)

        augmented_dataset = TensorDataset(augmented_images_tensor, augmented_labels_tensor)
        combined_dataset = ConcatDataset([train_dataset, augmented_dataset])
    else:
        combined_dataset = train_dataset

    # Verify the new class distribution
    combined_class_distribution = get_class_distribution(combined_dataset)
    print("Combined class distribution:", combined_class_distribution)
    logger.info(f"Combined class distribution: {combined_class_distribution}")
    return combined_dataset

def calculate_metrics_multiclass(gt_masks, pred_masks, num_classes=3):
    gt_masks = gt_masks.float()
    pred_masks = pred_masks.float()
    metrics = torch.zeros((num_classes, 4))

    for c in range(num_classes):
        gt_class = (gt_masks == c).float()
        pred_class = (pred_masks == c).float()
        tp = torch.sum(gt_class * pred_class)
        tn = torch.sum((1 - gt_class) * (1 - pred_class))
        fp = torch.sum((1 - gt_class) * pred_class)
        fn = torch.sum(gt_class * (1 - pred_class))
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        metrics[c] = torch.tensor([accuracy.item(), precision.item(), recall.item(), f1_score.item()])

    acc = torch.sum(gt_masks == pred_masks) / len(pred_masks)
    return acc, metrics


def return_model(model_num = 0):
    if(model_num == 4):
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        print("learning rate: ", lr)
        for param in model.parameters():
            param.requires_grad = False
        model.heads.head = nn.Linear(768, 3, bias=True)
        return model
    
    if(model_num == 3):
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        model = nn.Sequential(collections.OrderedDict([
          ('dino', dino),
          ('last', nn.Linear(384, 3, bias=True)),
        ]))
        for param in model.parameters():
            param.requires_grad = False
        model.last = nn.Linear(384, 3, bias=True)
        return model
    
    if(model_num == 2):
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, 3, bias=True) 
        return model
        
    if(model_num == 1):
        return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    else:
        return models.resnet18()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device)
torch.set_grad_enabled(False)
hidden = 384

X_data = torch.zeros((len(train_dataset), hidden), requires_grad=False)
b_data = torch.zeros((len(train_dataset), 3), requires_grad=False)

for k, (y, x) in enumerate(train_dataloader2):
    if(k % 250 == 0):
        print(k)
    X_data[k, :] = model(y.to(device).repeat(1, 3, 1, 1))
    xn = torch.zeros(1, 3).to(device)
    xn[:, x] = 1
    b_data[k, :] = xn

names = ["Healthy", "Bacterial P.", "Viral P."]
# Convert one-hot to class indices
class_labels = np.argmax(b_data, axis=1)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=40, perplexity=30, init='pca', verbose=1)
tsne_results = tsne.fit_transform(X_data)

# Plot
plt.figure(figsize=(9, 6))
for i in range(3):  # 3 classes
    idx = class_labels == i
    plt.scatter(tsne_results[idx, 0]/10, tsne_results[idx, 1]/10, label=names[i], alpha=0.7)

plt.legend(fontsize=17, handletextpad=0.5, borderaxespad=0.2, loc='upper right')
#plt.legend(fontsize=17)
plt.xlabel('t-SNE 1', fontsize=22)
plt.ylabel('t-SNE 2', fontsize=22)
# Increase tick label sizes
plt.tick_params(axis='both', which='major', labelsize=18)
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('tsne_plot2.png', dpi=300)
plt.close()
