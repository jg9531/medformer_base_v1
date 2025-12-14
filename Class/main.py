import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import logging
import wandb  # For experiment tracking

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from pathlib import Path

import medmnist
from medmnist import INFO
from engine import train, evaluate
from losses import *  # Custom loss functions
from m import *


# ---------------------------- Setup Logging ----------------------------
def setup_logging(save_dir):
    """Setup logging configuration"""
    log_file = save_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ---------------------------- Argument Parser ----------------------------
def get_args():

    parser = argparse.ArgumentParser(description=' Training Configuration')

    # Dataset
    parser.add_argument('--data_flag', type=str, required=True, help='MedMNIST dataset flag')
    parser.add_argument('--data_path', type=str, required=True, help='Path to store datasets')

    # Training arguments
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, required=True, help='Warmup epochs')
    parser.add_argument('--early_stopping_patience', type=int, required=True, help='Early stopping patience')

    # Model arguments
    parser.add_argument('--model_type', type=str, required=True, choices=['m_vit_small', 'm_vit_base', 'm_vit_large', 'm', 'm_heavy'], help='model')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout rate')

    # Loss and optimization
    parser.add_argument('--loss', type=str, required=True, choices=['ce', 'smooth_ce', 'focal', 'bce', 'weighted_ce'], help='Loss function')
    parser.add_argument('--optimizer', type=str, required=True, choices=['sgd', 'adam', 'adamw'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, required=True, choices=['none', 'step', 'cosine', 'plateau'], help='Learning rate scheduler')

    # Data augmentation
    parser.add_argument('--image_size', type=int, required=True, help='Input image size')
    parser.add_argument('--augmentation', type=str, required=True, choices=['none', 'basic', 'strong'], help='Augmentation strategy')
    parser.add_argument('--mixup_alpha', type=float, required=True, help='Mixup alpha (0 to disable)')
    parser.add_argument('--cutmix_alpha', type=float, required=True, help='CutMix alpha (0 to disable)')

    # Regularization
    parser.add_argument('--label_smoothing', type=float, required=True, help='Label smoothing factor')
    parser.add_argument('--gradient_clip', type=float, required=True, help='Gradient clipping norm')

    # Validation and testing
    parser.add_argument('--val_frequency', type=int, required=True, help='Validation frequency (epochs)')
    parser.add_argument('--save_frequency', type=int, required=True, help='Model save frequency (epochs)')

    # Paths and experiment tracking
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, required=True, help='W&B project name')

    # Hardware
    parser.add_argument('--device', type=str, required=True, help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for data loading')

    # Testing and evaluation
    parser.add_argument('--test_only', action='store_true', help='Only run testing')
    parser.add_argument('--save_predictions', action='store_true', help='Save model predictions')
    parser.add_argument('--compute_metrics', action='store_true', help='Compute detailed metrics')

    args = parser.parse_args()

    
    return parser.parse_args()


# ---------------------------- Data Augmentation Strategies ----------------------------
def get_transforms(args, split='train'):
    """Get data transforms based on augmentation strategy"""
    base_size = args.image_size
    
    if split == 'train':
        if args.augmentation == 'none':
            transform = transforms.Compose([
                transforms.Resize((base_size, base_size)),
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif args.augmentation == 'basic':
            transform = transforms.Compose([
                transforms.Resize((base_size + 32, base_size + 32)),
                transforms.RandomCrop((base_size, base_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif args.augmentation == 'strong':
            transform = transforms.Compose([
                transforms.Resize((base_size + 32, base_size + 32)),
                transforms.RandomCrop((base_size, base_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:  # validation/test
        transform = transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


# ---------------------------- Model Factory ----------------------------
def create_model(args, n_classes, device):
    """Create model based on arguments"""
    if args.model_type == 'm_vit_small':
        model = m_vit(
            stem_chs=[64, 32, 64], 
            depths=[3, 4, 10, 3], 
            path_dropout=args.dropout,
            num_classes=n_classes
        )
    elif args.model_type == 'm_vit_base':
        model = m_vit(
            stem_chs=[64, 32, 64], 
            depths=[3, 4, 20, 3], 
            path_dropout=args.dropout,
            num_classes=n_classes
        )
    elif args.model_type == 'm_vit_large':
        model = m_vit(
            stem_chs=[64, 32, 64], 
            depths=[3, 4, 30, 3], 
            path_dropout=args.dropout,
            num_classes=n_classes
        )
    elif args.model_type == 'm':
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
        )
    elif args.model_type == 'm_heavy':
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
        )
    
    return model.to(device)


# ---------------------------- Loss Function Factory ----------------------------
def create_criterion(args, task, n_classes, train_dataset=None):
    """Create loss function based on arguments"""
    if task == "multi-label, binary-class":
        return get_bce_with_logits()
    
    if args.loss == 'ce':
        if args.label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        else:
            return nn.CrossEntropyLoss()
    elif args.loss == 'smooth_ce':
        return LabelSmoothingCrossEntropy(smoothing=0.1)
    elif args.loss == 'focal':
        return FocalLoss(gamma=2.0)
    elif args.loss == 'bce':
        return get_bce_with_logits()
    elif args.loss == 'weighted_ce':
        # Calculate class weights from training data
        if train_dataset is not None:
            class_counts = torch.zeros(n_classes)
            for _, label in train_dataset:
                class_counts[label] += 1
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * n_classes
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")


# ---------------------------- Optimizer and Scheduler Factory ----------------------------
def create_optimizer_and_scheduler(args, model, steps_per_epoch):
    """Create optimizer and learning rate scheduler"""
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=0.9, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    
    # Scheduler
    if args.scheduler == 'none':
        scheduler = None
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    return optimizer, scheduler


# ---------------------------- Checkpoint Utilities ----------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_path, args):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'args': vars(args)
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('best_acc', 0.0)


# ---------------------------- Training Loop with Advanced Features ----------------------------
def train_epoch_advanced(model, train_loader, criterion, optimizer, device, epoch, args, logger):
    """Advanced training loop with gradient clipping, mixup, etc."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Apply Mixup or CutMix if enabled
        if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
            # Implementation would go here
            pass
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.squeeze().long())
        
        loss.backward()
        
        # Gradient clipping
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target.squeeze().long()).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    logger.info(f'Train Epoch {epoch}: Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


# ---------------------------- Main Script ----------------------------
def main():
    args = get_args()
    
    # Setup experiment directory
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_type}_{args.data_flag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir)
    
    # Save arguments
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup W&B if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Device setup
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Dataset setup
    data_flag = args.data_flag
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    
    logger.info(f"Dataset: {data_flag}")
    logger.info(f"Task: {task}")
    logger.info(f"Number of channels: {n_channels}")
    logger.info(f"Number of classes: {n_classes}")
    
    # Data transforms
    train_transform = get_transforms(args, 'train')
    test_transform = get_transforms(args, 'test')
    
    # Load datasets
    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    val_dataset = DataClass(split='val' if 'val' in DataClass.flag else 'test', 
                           transform=test_transform, download=True)
    test_dataset = DataClass(split='test', transform=test_transform, download=True)
    
    # Data loaders
    train_loader = data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    val_loader = data.DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # Model setup
    model = create_model(args, n_classes, device)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = create_criterion(args, task, n_classes, train_dataset)
    
    # Optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(args, model, len(train_loader))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_epoch} with best accuracy {best_acc:.2f}%")
    
    # Test only mode
    if args.test_only:
        logger.info("Running test only...")
        test_results = evaluate(model, test_loader, data_flag=data_flag, split='test', device=device)
        logger.info(f"Test results: {test_results}")
        return
    
    # Training loop
    early_stopping_counter = 0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Training
        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device, epoch + 1, args, logger
        )
        
        # Validation
        if (epoch + 1) % args.val_frequency == 0:
            val_results = evaluate(model, val_loader, data_flag=data_flag, split='val', device=device)
            val_acc = val_results.get('accuracy', 0.0)
            val_loss = val_results.get('loss', 0.0)
            
            logger.info(f'Validation - Epoch {epoch + 1}: Loss: {val_loss:.6f}, Accuracy: {val_acc:.2f}%')
            
            # Update learning rate scheduler
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, best_acc,
                    save_dir / 'best_model.pth', args
                )
                early_stopping_counter = 0
                logger.info(f'New best model saved with accuracy: {best_acc:.2f}%')
            else:
                early_stopping_counter += 1
            
            # Update training history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # W&B logging
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        # Save periodic checkpoints
        if (epoch + 1) % args.save_frequency == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_acc,
                save_dir / f'checkpoint_epoch_{epoch + 1}.pth', args
            )
        
        # Early stopping
        if early_stopping_counter >= args.early_stopping_patience:
            logger.info(f'Early stopping at epoch {epoch + 1}')
            break
    
    # Final evaluation on test set
    logger.info("Final evaluation on test set...")
    # Load best model
    load_checkpoint(save_dir / 'best_model.pth', model)
    test_results = evaluate(model, test_loader, data_flag=data_flag, split='test', device=device)
    logger.info(f"Final test results: {test_results}")
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    
    if args.use_wandb:
        wandb.finish()
    
    logger.info(f"Training completed. Results saved to {save_dir}")


if __name__ == '__main__':
    main()