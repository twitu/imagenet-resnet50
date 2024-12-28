import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import tarfile
from PIL import ImageFile, Image
import io
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageNetFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_info = []
        self.classes = set()
        
        # Scan directory for images
        for filename in os.listdir(root_dir):
            if filename.endswith(('.JPEG', '.jpg', '.jpeg')):
                # Parse class ID from filename (e.g., 'n01644373' from ILSVRC2012_val_00017505_n01644373.JPEG)
                class_name = filename.split('_')[-1].split('.')[0]
                self.classes.add(class_name)
                self.image_info.append(filename)
        
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_name = self.image_info[idx]
        image_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Get class name from filename
        class_name = img_name.split('_')[-1].split('.')[0]
        label = self.class_to_idx[class_name]
        return image, label

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
        
        self.test_transform = transforms.Compose([
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
            test_dir = os.path.join(data_path, 'test')
            
            print("Loading training dataset...")
            train_dataset = ImageNetFolder(
                root_dir=train_dir,
                transform=self.train_transform
            )
            
            print("Loading validation dataset...")
            val_dataset = ImageNetFolder(
                root_dir=val_dir,
                transform=self.val_transform
            )
            
            print("Loading test dataset...")
            test_dataset = ImageNetFolder(
                root_dir=test_dir,
                transform=self.test_transform
            )
            
            num_workers = min(8, os.cpu_count())
            
            # Create all three loaders
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
            
            self.test_loader = DataLoader(
                test_dataset,
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
            print(f"Number of test samples: {len(test_dataset)}")
            print(f"Number of classes: {len(train_dataset.classes)}")
            print(f"Using {num_workers} workers for data loading")
            
            return self.train_loader, self.val_loader, self.test_loader
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise