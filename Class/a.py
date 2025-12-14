import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

import medmnist
from medmnist import INFO
from engine import train, evaluate
from losses import *  # Custom loss functions
from MedViT import *


# ---------------------------- Argument Parser ----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="MedViT Training Script")
    parser.add_argument('--data_flag', type=str, default='retinamnist', help='MedMNIST dataset flag')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='small', choices=['m_vit_small', 'm_vit_base', 'm_vit_large', 'm', 'm_heavy'], help='MedViT model size')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'smooth_ce', 'focal', 'bce'], help='Loss function')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'], help='Optimizer to use')
    parser.add_argument('--image_size', type=int, default=336, help='Input image size (height and width)')
    return parser.parse_args()

# ---------------------------- Main Script ----------------------------
def main():
    args = get_args()

    data_flag = args.data_flag
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    lr = args.lr
    model_type = args.model_type
    loss_choice = args.loss
    optimizer_choice = args.optimizer
    download = True

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    print("Number of channels:", n_channels)
    print("Number of classes :", n_classes)

    # ---------------------------- Transforms ----------------------------
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # ---------------------------- Load Data ----------------------------
    train_dataset = DataClass(split='train', transform=train_transform, download=download)
    test_dataset = DataClass(split='test', transform=test_transform, download=download)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)

    print("Train Dataset:", train_dataset)
    print("===================")
    print("Test Dataset :", test_dataset)

    # ---------------------------- Model Setup ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'm_vit_small':
        model = MedViT(stem_chs=[64, 32, 64], 
                depths=[3, 4, 10, 3], 
                path_dropout=0.1).to(device)
    elif model_type == 'm_vit_base':
        model = MedViT(stem_chs=[64, 32, 64], 
                depths=[3, 4, 20, 3], 
                path_dropout=0.2).to(model)
    elif model_type == 'm_vit_large':
        model = MedViT(stem_chs=[64, 32, 64], 
                depths=[3, 4, 30, 3], 
                path_dropout=0.2).to(device)
    elif model_type == 'm':
        model = SwinTransformer(
            hidden_dim=96,
            layers=(2, 2, 6, 2, 2),
            heads=(3, 6, 12, 24, 48),
            channels=3,
            num_classes=n_classes,
            head_dim=32,
            window_size=3,
            downscaling_factors=(2, 2, 2, 2, 2),
            relative_pos_embedding=True
        ).to(device)

    elif model_type == 'm_heavy':
        model = SwinTransformer1(
            hidden_dim=96,
            layers=(2, 2, 6, 2, 2),
            heads=(3, 6, 12, 24, 48),
            channels=3,
            num_classes=n_classes,
            head_dim=32,
            window_size=3,
            downscaling_factors=(2, 2, 2, 2, 2),
            relative_pos_embedding=True
        ).to(device)

    # ---------------------------- Loss Function ----------------------------
    if task == "multi-label, binary-class":
        criterion = get_bce_with_logits()
    else:
        if loss_choice == 'ce':
            criterion = get_cross_entropy()
        elif loss_choice == 'smooth_ce':
            criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        elif loss_choice == 'focal':
            criterion = FocalLoss(gamma=2.0)
        elif loss_choice == 'bce':
            criterion = get_bce_with_logits()
        else:
            raise ValueError(f"Unsupported loss: {loss_choice}")

    # ---------------------------- Optimizer Setup ----------------------------
    if optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_choice}")

    # ---------------------------- Train & Evaluate ----------------------------
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        train(model, train_loader, criterion, optimizer, task, device)

    evaluate(model, test_loader, data_flag=data_flag, split='test', device=device)

if __name__ == '__main__':
    main()
