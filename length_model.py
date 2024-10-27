import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import time
import argparse

# Custom Dataset class remains the same
class CaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# CNN Model remains the same
class CaptchaLengthNet(nn.Module):
    def __init__(self):
        super(CaptchaLengthNet, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fourth Convolutional Block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 12 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

def prepare_data(base_path):
    print("Preparing dataset...")
    image_paths = []
    labels = []
    
    for length in range(2, 7):
        folder_path = os.path.join(base_path, f'train{length}')
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(files)} images in train{length}/")
            for img_name in files:
                image_paths.append(os.path.join(folder_path, img_name))
                labels.append(length - 2)
    
    if not image_paths:
        raise ValueError(f"No images found in {base_path}. Please check the directory structure.")
    
    print(f"Total images found: {len(image_paths)}")
    return image_paths, labels

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('best_val_acc', 0)
    return 0, 0

def train_model(base_path, num_epochs=10, batch_size=32, learning_rate=0.001, resume=None):
    # Set device and enable cudnn benchmarking for better performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    
    try:
        # Prepare data
        image_paths, labels = prepare_data(base_path)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets and dataloaders
        train_dataset = CaptchaDataset(X_train, y_train, transform=transform)
        val_dataset = CaptchaDataset(X_val, y_val, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              num_workers=4, pin_memory=True)

        # Initialize model, loss function, and optimizer
        model = CaptchaLengthNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   patience=2, factor=0.5, verbose=True)

        # Load checkpoint if resume is provided
        start_epoch = 0
        best_val_acc = 0
        if resume:
            start_epoch, best_val_acc = load_checkpoint(model, optimizer, resume)
            print(f"Resumed training from epoch {start_epoch+1} with best validation accuracy {best_val_acc:.2f}%")

        # Training loop
        early_stopping_counter = 0
        early_stopping_patience = 5
        start_time = time.time()
        
        print("\nStarting training...")
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for images, labels in train_pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'loss': f'{train_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_pbar.set_postfix({
                        'loss': f'{val_loss/val_total:.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })

            # Calculate epoch statistics
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * correct / total
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
            print(f'Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_captcha_model.pth')
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            # Early stopping check
            if early_stopping_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
            
            print('-' * 60)

        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/60:.2f} minutes')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        return model
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CAPTCHA length classifier')
    parser.add_argument('--base_path', type=str, default='./',
                        help='Base path containing training data directories')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Train the model with command line arguments
    try:
        model = train_model(
            base_path=args.base_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            resume=args.resume
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")
