from model import ImageNetModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from tqdm import tqdm
import torch
from train import DataLoader_ImageNet
from torch.cuda.amp import autocast, GradScaler

def train(model, device, train_loader, optimizer, epoch, scaler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    running_loss = 0.0
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Runs the forward pass with autocasting
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Scales loss and calls backward() to create scaled gradients
        scaler.scale(loss).backward()
        # Unscales gradients and calls or skips optimizer.step()
        scaler.step(optimizer)
        # Updates the scale for next iteration
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss/len(train_loader)})

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss/len(test_loader):.3f}, Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    # Hyperparameters
    EPOCHS = 50
    LR = 0.001
    WEIGHT_DECAY = 0.05
    
    # Initialize model
    model_wrapper = ImageNetModel()
    model = model_wrapper.model
    device = model_wrapper.device
    
    # Print model summary
    model_wrapper.get_model_summary()
    
    # Initialize data loader with verification
    data_loader = DataLoader_ImageNet()
    train_loader, val_loader = data_loader.load_data('/path/to/imagenet')
    
    # Verify data loaders
    try:
        # Check if we can get a batch
        sample_batch, sample_labels = next(iter(train_loader))
        print(f"Successfully loaded batch of shape: {sample_batch.shape}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        # Verify data is moving to GPU if available
        sample_batch = sample_batch.to(device)
        print(f"Data successfully moved to device: {sample_batch.device}")
        
    except Exception as e:
        print(f"Error in data loading verification: {str(e)}")
        raise
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEPOCH: {epoch+1}/{EPOCHS}")
        train(model, device, train_loader, optimizer, epoch, scaler)
        scheduler.step()
        
        # Test doesn't need mixed precision
        accuracy = test(model, device, val_loader)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, 'best_model.pth')

if __name__ == "__main__":
    main() 