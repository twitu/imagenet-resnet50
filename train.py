import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataLoader_ImageNet:
    def __init__(self, batch_size=256):
        self.batch_size = batch_size if torch.cuda.is_available() else 32
        
        # Enhanced transforms for ImageNet
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
    def load_data(self, data_path):
        try:
            train_dir = os.path.join(data_path, 'train')
            val_dir = os.path.join(data_path, 'val')
            
            # Add more detailed directory checking
            print(f"Checking directory structure...")
            print(f"Train directory exists: {os.path.exists(train_dir)}")
            print(f"Val directory exists: {os.path.exists(val_dir)}")
            
            if os.path.exists(train_dir):
                sample_classes = os.listdir(train_dir)[:5]
                print(f"Sample training classes found: {sample_classes}")
                
            if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
                raise FileNotFoundError(
                    f"Training or validation directory not found in {data_path}. "
                    f"Expected {train_dir} and {val_dir}"
                )
            
            print(f"Loading training data from: {train_dir}")
            train_dataset = datasets.ImageFolder(
                root=train_dir,
                transform=self.train_transform
            )
            
            print(f"Loading validation data from: {val_dir}")
            val_dataset = datasets.ImageFolder(
                root=val_dir,
                transform=self.val_transform
            )
            
            num_workers = min(8, os.cpu_count())
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
            
            print(f"Dataset loaded successfully!")
            print(f"Number of training samples: {len(train_dataset)}")
            print(f"Number of validation samples: {len(val_dataset)}")
            print(f"Number of classes: {len(train_dataset.classes)}")
            print(f"Using {num_workers} workers for data loading")
            
            return self.train_loader, self.val_loader
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise