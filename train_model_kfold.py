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
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset, ConcatDataset
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet101, ResNet101_Weights
from torchvision.models import vit_l_16, ViT_L_16_Weights, vit_b_16, ViT_B_16_Weights

from collections import namedtuple, Counter, defaultdict
import torch.nn.functional as F
from torch.nn.functional import relu
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from sklearn.model_selection import KFold

from IPython.core.debugger import Pdb
import sys
import cv2 as cv
import pickle
import collections
import logging

task_name = "resnet_transferlearn" # change for each experiment    
logger_folder = "loggers_aug_last"
logger_folder_name = logger_folder+"/"+task_name
save_dir = logger_folder_name+"/experiment"

if not os.path.exists(logger_folder):
    os.mkdir(logger_folder)
if not os.path.exists(logger_folder_name):
    os.mkdir(logger_folder_name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | | %(levelname)s | | %(message)s')

logger_file_name = os.path.join(logger_folder_name, "experiment")
file_handler = logging.FileHandler(logger_file_name,'w')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.info('Code started \n')


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

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=(3, 3)),
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

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device).repeat(1, 3, 1, 1)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)

        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).repeat(1, 3, 1, 1)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())
        all_preds_tensor = torch.tensor(all_preds)
        all_labels_tensor = torch.tensor(all_labels)
        acc, metrics = calculate_metrics_multiclass(all_labels_tensor, all_preds_tensor)
        if torch.mean(metrics[:, 3]) > best_f1:
            best_f1 = torch.mean(metrics[:, 3])
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Val Acc: {acc:.4f}, Val Precision: {torch.mean(metrics[:, 1]):.4f}, "
              f"Val Recall: {torch.mean(metrics[:, 2]):.4f}, Val F1: {torch.mean(metrics[:, 3]):.4f}, Best F1: {best_f1:.4f}")
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Val Acc: {acc:.4f}, Val Precision: {torch.mean(metrics[:, 1]):.4f}, "
              f"Val Recall: {torch.mean(metrics[:, 2]):.4f}, Val F1: {torch.mean(metrics[:, 3]):.4f}, Best F1: {best_f1:.4f}")
        
    model.load_state_dict(best_model_wts)
    return model
        
def evaluate_model_x(d1, d2, model):
    datatype = torch.float64
    model.eval()
    print("")
    logger.info(f"")
    sets = ["train", "test"]
    for idx, dataset in enumerate([d1, d2]):
        ct = 0 
        all_labels = []
        all_preds = []
        for inputs, labels in dataset:
            ct+=1
            inputs = inputs.to(device).repeat(1, 3, 1, 1)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())
                #Pdb().set_trace() 
        all_preds_tensor = torch.tensor(all_preds)
        all_labels_tensor = torch.tensor(all_labels)
        acc, metrics = calculate_metrics_multiclass(all_labels_tensor, all_preds_tensor)
        print(f'Average Metrics over {sets[idx]} dataset:')
        print(f'Accuracy: {acc:.4f}')
        print(f'Precision: {torch.mean(metrics[:, 1]):.4f}')
        print(f'Recall: {torch.mean(metrics[:, 2]):.4f}')
        print(f'F1 Score: {torch.mean(metrics[:, 3]):.4f}')
        logger.info(f'Average Metrics over {sets[idx]} dataset:')
        logger.info(f'Accuracy: {torch.mean(metrics[:, 0]):.4f}')
        logger.info(f'Precision: {torch.mean(metrics[:, 1]):.4f}')
        logger.info(f'Recall: {torch.mean(metrics[:, 2]):.4f}')
        logger.info(f'F1 Score: {torch.mean(metrics[:, 3]):.4f}')

        for c in range(metrics.shape[0]):
            print(f'\nClass {c} Metrics')
            print(f'Accuracy: {metrics[c, 0]:.4f}')
            print(f'Precision: {metrics[c, 1]:.4f}')
            print(f'Recall: {metrics[c, 2]:.4f}')
            print(f'F1 Score: {metrics[c, 3]:.4f}')
            logger.info(f'\nClass {c} Metrics')
            logger.info(f'Accuracy: {metrics[c, 0]:.4f}')
            logger.info(f'Precision: {metrics[c, 1]:.4f}')
            logger.info(f'Recall: {metrics[c, 2]:.4f}')
            logger.info(f'F1 Score: {metrics[c, 3]:.4f}')
        print()
        print()
        logger.info(f"")
        logger.info(f"")
        

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
    
    
# K-fold Cross Validation
def k_fold_cross_validation(model_no, train_dataset, num_folds=5, num_epochs=30, lr=0.001, wd=0.0001, bs=16):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_metrics = []
    fold_acc = []
    best_model = None
    best_f1 = 0

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold + 1}')
        logger.info(f'Fold {fold + 1}')
        train_subsampler = Subset(train_dataset, train_ids)
        val_subsampler = Subset(train_dataset, val_ids)
        
        train_subsampler = augment_dataset(train_subsampler)
        train_loader = DataLoader(train_subsampler, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=bs, shuffle=False)

        #model = models.resnet18()
        model = return_model(model_no)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        print(f"- learning rate: {lr}")
        print(f"- weight decay:  {wd}")
        print(f"- batch size:    {bs}")
        logger.info(f"- learning rate: {lr}")
        logger.info(f"- weight decay:  {wd}")
        logger.info(f"- batch size:    {bs}")

        trained_model = train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs)

        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).repeat(1, 3, 1, 1)
                labels = labels.to(device)
                outputs = trained_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())
        all_preds_tensor = torch.tensor(all_preds)
        all_labels_tensor = torch.tensor(all_labels)
        acc, metrics = calculate_metrics_multiclass(all_labels_tensor, all_preds_tensor)
        fold_metrics.append(metrics)
        fold_acc.append(acc)
        
        if torch.mean(metrics[:, 3]) > best_f1:
            best_f1 = torch.mean(metrics[:, 3])
            best_model = trained_model

    fold_metrics_tensor = torch.stack(fold_metrics)
    mean_metrics = torch.mean(fold_metrics_tensor, dim=0)
    std_metrics = torch.std(fold_metrics_tensor, dim=0)
    fold_acc = torch.tensor(fold_acc)

    print(f'Average Metrics over {num_folds} folds:')
    print(f'Accuracy: {torch.mean(fold_acc):.4f} ± {torch.std(fold_acc):.4f}')
    print(f'Precision: {torch.mean(mean_metrics[:, 1]):.4f} ± {torch.mean(std_metrics[:, 1]):.4f}')
    print(f'Recall: {torch.mean(mean_metrics[:, 2]):.4f} ± {torch.mean(std_metrics[:, 2]):.4f}')
    print(f'F1 Score: {torch.mean(mean_metrics[:, 3]):.4f} ± {torch.mean(std_metrics[:, 3]):.4f}')
    logger.info(f'Average Metrics over {num_folds} folds:')
    logger.info(f'Accuracy: {torch.mean(fold_acc):.4f} ± {torch.std(fold_acc):.4f}')
    logger.info(f'Precision: {torch.mean(mean_metrics[:, 1]):.4f} ± {torch.mean(std_metrics[:, 1]):.4f}')
    logger.info(f'Recall: {torch.mean(mean_metrics[:, 2]):.4f} ± {torch.mean(std_metrics[:, 2]):.4f}')
    logger.info(f'F1 Score: {torch.mean(mean_metrics[:, 3]):.4f} ± {torch.mean(std_metrics[:, 3]):.4f}')

    for c in range(mean_metrics.shape[0]):
        print(f'\nClass {c} Metrics over {num_folds} folds:')
        print(f'Accuracy: {mean_metrics[c, 0]:.4f} ± {std_metrics[c, 0]:.4f}')
        print(f'Precision: {mean_metrics[c, 1]:.4f} ± {std_metrics[c, 1]:.4f}')
        print(f'Recall: {mean_metrics[c, 2]:.4f} ± {std_metrics[c, 2]:.4f}')
        print(f'F1 Score: {mean_metrics[c, 3]:.4f} ± {std_metrics[c, 3]:.4f}')
        logger.info(f'\nClass {c} Metrics over {num_folds} folds:')
        logger.info(f'Accuracy: {mean_metrics[c, 0]:.4f} ± {std_metrics[c, 0]:.4f}')
        logger.info(f'Precision: {mean_metrics[c, 1]:.4f} ± {std_metrics[c, 1]:.4f}')
        logger.info(f'Recall: {mean_metrics[c, 2]:.4f} ± {std_metrics[c, 2]:.4f}')
        logger.info(f'F1 Score: {mean_metrics[c, 3]:.4f} ± {std_metrics[c, 3]:.4f}')

    average_f1_score = torch.mean(mean_metrics[:, 3]).item()
    return trained_model, average_f1_score


num_epochs = 25
num_folds_k = 5
best_model = None
best_average_f1_score = 0

for learning_rate in [0.001, 0.0005, 0.0001]:
    for weight_decay in [1e-4, 1e-5]:
        for batch_size in [16, 32]:
            model, average_f1_score = k_fold_cross_validation(1, train_dataset, num_folds=num_folds_k, num_epochs=num_epochs, lr=learning_rate, wd=weight_decay, bs=batch_size)
            evaluate_model_x(train_dataloader2, test_dataloader, model)
            if average_f1_score > best_average_f1_score:
                best_average_f1_score = average_f1_score
                best_model = model
                with open(logger_folder_name+'/resnet_transferlearn.pkl', 'wb') as f:
                    pickle.dump(best_model, f)

print(f'Best Average F1 Score: {best_average_f1_score:.4f}')
logger.info(f'Best Average F1 Score: {best_average_f1_score:.4f}')


with open(logger_folder_name+'/resnet_transferlearn.pkl', 'rb') as f:
    best_model = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(device)
evaluate_model_x(train_dataloader2, test_dataloader, best_model)