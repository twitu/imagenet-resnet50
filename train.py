import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import tarfile
from PIL import ImageFile, Image
import io
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TarImageFolder(Dataset):
    def __init__(self, tar_files, transform=None):
        self.tar_files = tar_files if isinstance(tar_files, list) else [tar_files]
        self.transform = transform
        self.image_list = []
        
        # Extract image info from all tar files
        for tar_path in self.tar_files:
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith(('.JPEG', '.jpg', '.jpeg')):
                        # Extract class from directory structure
                        class_name = member.name.split('/')[0]
                        self.image_list.append({
                            'tar_path': tar_path,
                            'member_name': member.name,
                            'class': class_name
                        })
        
        # Create class mapping
        self.classes = sorted(list(set(info['class'] for info in self.image_list)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_info = self.image_list[idx]
        
        # Open tar file and extract image
        with tarfile.open(image_info['tar_path'], 'r:gz') as tar:
            member = tar.getmember(image_info['member_name'])
            f = tar.extractfile(member)
            image_data = f.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[image_info['class']]
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
            # Collect all tar files
            train_tars = sorted([
                os.path.join(data_path, f) for f in os.listdir(data_path)
                if f.startswith('train_images') and f.endswith('.tar.gz')
            ])
            
            val_tar = os.path.join(data_path, 'val_images.tar.gz')
            test_tar = os.path.join(data_path, 'test_images.tar.gz')
            
            print("Found training files:", train_tars)
            print("Found validation file:", val_tar)
            print("Found test file:", test_tar)
            
            # Create datasets
            print("Loading training dataset...")
            train_dataset = TarImageFolder(
                tar_files=train_tars,
                transform=self.train_transform
            )
            
            print("Loading validation dataset...")
            val_dataset = TarImageFolder(
                tar_files=val_tar,
                transform=self.val_transform
            )
            
            print("Loading test dataset...")
            test_dataset = TarImageFolder(
                tar_files=test_tar,
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