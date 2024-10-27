import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import argparse
import random
import logging
from datetime import datetime
import json

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, symbols_file, transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.samples = []

        # Read symbols file
        with open(symbols_file, 'r') as f:
            self.symbols = f.readline().strip()
            
        # Log the symbols for debugging
        logging.info(f"Loaded symbols: {self.symbols}")
        logging.info(f"Number of symbols: {len(self.symbols)}")
        
        # Create symbol to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.symbols)}
        
        # Debug print the mapping
        logging.info("Character to index mapping:")
        for char, idx in self.char_to_idx.items():
            logging.info(f"'{char}' -> {idx}")
        
        # Compile statistics for analysis
        length_distribution = {}
        skipped_files = []
        valid_files = []
        
        # Track unique labels to handle multiple versions of same captcha
        unique_labels = set()
        
        for img_file in os.listdir(data_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                # Remove the file extension
                base_name = os.path.splitext(img_file)[0]
                
                # Extract label by removing the _N suffix pattern
                # This will handle _1, _2, _3, etc.
                parts = base_name.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    label = ''.join(parts[:-1])  # Join all parts except the last number
                else:
                    label = base_name
                
                # Validate the label
                if not label:
                    skipped_files.append((img_file, "Empty label after processing"))
                    continue
                
                # Check if all characters are valid
                invalid_chars = [c for c in label if c not in self.symbols]
                if invalid_chars:
                    skipped_files.append((img_file, f"Invalid characters: {invalid_chars}"))
                    continue
                
                # Store the sample
                self.samples.append((os.path.join(data_dir, img_file), label))
                valid_files.append((img_file, label))
                unique_labels.add(label)
                
                # Track length distribution
                length = len(label)
                length_distribution[length] = length_distribution.get(length, 0) + 1
        
        # Comprehensive logging
        logging.info(f"\n{'='*50}")
        logging.info(f"Dataset Analysis for: {data_dir}")
        logging.info(f"{'='*50}")
        logging.info(f"Total files found: {len(valid_files) + len(skipped_files)}")
        logging.info(f"Valid samples: {len(valid_files)}")
        logging.info(f"Unique captchas: {len(unique_labels)}")
        logging.info(f"Skipped samples: {len(skipped_files)}")
        
        # Log length distribution
        logging.info("\nCaptcha Length Distribution:")
        for length, count in sorted(length_distribution.items()):
            logging.info(f"Length {length}: {count} samples ({count/(len(valid_files))*100:.1f}%)")
        
        # Log sample of valid files
        logging.info("\nSample of valid files:")
        for file, label in valid_files[:5]:
            logging.info(f"  {file} -> label: '{label}' (length: {len(label)})")
        
        # Log distribution of suffixes for a few samples
        logging.info("\nExample of multiple versions of same captcha:")
        sample_labels = list(unique_labels)[:3]  # Take first 3 unique labels
        for label in sample_labels:
            matching_files = [f for f, l in valid_files if l == label]
            if len(matching_files) > 1:
                logging.info(f"  Label '{label}' appears in files: {matching_files}")
        
        if not self.samples:
            raise RuntimeError("No valid samples found in the dataset!")
        
        # # Read symbols file
        # with open(symbols_file, 'r') as f:
        #     self.symbols = f.readline().strip()
        
        # # Create symbol to index mapping
        # self.char_to_idx = {char: idx for idx, char in enumerate(self.symbols)}
        
        # # Collect all image files
        # for img_file in os.listdir(data_dir):
        #     if img_file.endswith(('.png', '.jpg', '.jpeg')):
        #         # label = img_file.split('_')[0]
        #         label = os.path.splitext(img_file)[0]
                
        #         if all(c in self.symbols for c in label):
        #             self.samples.append((os.path.join(data_dir, img_file), label))
        
        # logging.info(f"{'Training' if 'train' in data_dir else 'Validation'} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def apply_augmentation(self, image):
        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            image = TF.adjust_brightness(image, factor)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            image = TF.adjust_contrast(image, factor)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            image = TF.rotate(image, angle)
        
        return image

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.augment:
            image = self.apply_augmentation(image)
            
        if self.transform:
            image = self.transform(image)
            
        label_indices = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return image, label_indices

class CaptchaModel(nn.Module):
    def __init__(self, num_chars, num_classes, backbone='resnet18', pretrained=True):
        super(CaptchaModel, self).__init__()
        
        self.backbone_name = backbone
        
        # Initialize backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_size = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_size = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_size = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Create separate output layers for each character
        self.char_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            for _ in range(num_chars)
        ])
        
        self._initialize_weights()
        
        # Print model summary
        logging.info("Model Architecture:")
        logging.info(self)
        logging.info(f"Total parameters: {sum(p.numel() for p in self.parameters())}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return [char_out(features) for char_out in self.char_outputs]

def load_checkpoint(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('best_accuracy', 0)
    return 0, 0

def train_model(model, train_loader, valid_loader, device, args):
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), 
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.restart_epochs,
        T_mult=2,
        eta_min=args.min_lr
    )
    
    start_epoch = 0
    best_accuracy = 0
    
    # Load checkpoint if provided
    if args.resume:
        start_epoch, best_accuracy = load_checkpoint(model, args.resume)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'learning_rates': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        logging.info(f'\nEpoch {epoch+1}/{args.epochs}')
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Current learning rate: {current_lr:.6f}')
        
        # Training phase
        model.train()
        train_loss = 0
        correct_chars = 0
        total_chars = 0
        
        train_pbar = tqdm(train_loader, desc='Training')
        for images, labels in train_pbar:
            images = images.to(device)
            labels = [label.to(device) for label in labels.t()]
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = sum(criterion(output, label) for output, label in zip(outputs, labels))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            for output, label in zip(outputs, labels):
                _, predicted = torch.max(output, 1)
                total_chars += label.size(0)
                correct_chars += (predicted == label).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{train_loss/(train_pbar.n+1):.4f}',
                'acc': f'{100.*correct_chars/total_chars:.2f}%'
            })
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * correct_chars / total_chars
        
        # Validation phase
        model.eval()
        valid_loss = 0
        correct_chars = 0
        total_chars = 0
        
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc='Validation')
            for images, labels in valid_pbar:
                images = images.to(device)
                labels = [label.to(device) for label in labels.t()]
                
                outputs = model(images)
                loss = sum(criterion(output, label) for output, label in zip(outputs, labels))
                valid_loss += loss.item()
                
                for output, label in zip(outputs, labels):
                    _, predicted = torch.max(output, 1)
                    total_chars += label.size(0)
                    correct_chars += (predicted == label).sum().item()
                
                valid_pbar.set_postfix({
                    'loss': f'{valid_loss/(valid_pbar.n+1):.4f}',
                    'acc': f'{100.*correct_chars/total_chars:.2f}%'
                })
        
        epoch_valid_loss = valid_loss / len(valid_loader)
        epoch_valid_acc = 100. * correct_chars / total_chars
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['valid_loss'].append(epoch_valid_loss)
        history['valid_acc'].append(epoch_valid_acc)
        history['learning_rates'].append(current_lr)
        
        # Save checkpoint
        if epoch_valid_acc > best_accuracy:
            best_accuracy = epoch_valid_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'history': history
            }
            
            # Save checkpoint
            torch.save(checkpoint, f'{args.model_name}_best.pth')
            logging.info(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        
        # Save training history
        with open(f'{args.model_name}_history.json', 'w') as f:
            json.dump(history, f)

def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--width', type=int, default=198, help='Input image width')
    parser.add_argument('--height', type=int, default=96, help='Input image height')
    parser.add_argument('--length', type=int, default=2, help='CAPTCHA length')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                      choices=['resnet18', 'resnet34', 'resnet50'],
                      help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', 
                      help='Use pretrained backbone')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                      help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                      help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                      help='Weight decay for optimizer')
    parser.add_argument('--clip_grad', type=float, default=1.0, 
                      help='Gradient clipping value')
    parser.add_argument('--restart_epochs', type=int, default=5, 
                      help='Epochs before learning rate restart')
    
    # Data parameters
    parser.add_argument('--train_data', type=str,
                      help='Training data directory',default='train')
    parser.add_argument('--valid_data', type=str,
                      help='Validation data directory',default='validation')
    parser.add_argument('--symbols', type=str,
                      help='Symbols file path',default='symbols.txt')
    parser.add_argument('--augment', action='store_true', 
                      help='Use data augmentation')
    
    # Other parameters
    parser.add_argument('--model_name', type=str,
                      help='Model name for saving',default='captcha_model2')
    parser.add_argument('--resume', type=str, default=None, 
                      help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'{args.model_name}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Read symbols file
    with open(args.symbols, 'r') as f:
        symbols = f.readline().strip()
    num_classes = len(symbols)
    logging.info(f'Number of symbols: {num_classes}')
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = CaptchaDataset(args.train_data, args.symbols, 
                                 transform, augment=args.augment)
    valid_dataset = CaptchaDataset(args.valid_data, args.symbols, 
                                 transform, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    # Create model
    model = CaptchaModel(args.length, num_classes, 
                        backbone=args.backbone,
                        pretrained=args.pretrained).to(device)
    
    # Train model
    train_model(model, train_loader, valid_loader, device, args)

if __name__ == '__main__':
    main()