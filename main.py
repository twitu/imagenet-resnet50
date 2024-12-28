from model import ImageNetModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from tqdm import tqdm
import torch
from train import DataLoader_ImageNet
from torch.cuda.amp import autocast, GradScaler

def train(model, device, train_loader, optimizer, scheduler, epoch, scaler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    running_loss = 0.0
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        avg_loss = running_loss/len(train_loader)
        pbar.set_postfix({'loss': avg_loss})
    
    # Step scheduler with average loss for the epoch
    scheduler.step(avg_loss)
    return avg_loss

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
    EPOCHS = 100
    LR = 0.01
    WEIGHT_DECAY = 0.05
    
    # Initialize model
    model_wrapper = ImageNetModel()
    model = model_wrapper.model
    device = model_wrapper.device
    
    # Print model summary
    model_wrapper.get_model_summary()
    
    # Initialize data loader with verification
    data_loader = DataLoader_ImageNet()
    train_loader, val_loader, test_loader = data_loader.load_data('/mnt/ebs_volume/data')
    
    # Verify data loaders
    try:
        print(f"Number of test batches: {len(test_loader)}")
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
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,
        patience=10,
        verbose=True,  # Print message when LR changes
        threshold=0.0001,
        threshold_mode='rel',
        cooldown=0,
        min_lr=0,
        eps=1e-08
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEPOCH: {epoch+1}/{EPOCHS}")
        train_loss = train(model, device, train_loader, optimizer, scheduler, epoch, scaler)
        
        # Test doesn't need mixed precision
        accuracy = test(model, device, val_loader)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
                'loss': train_loss,
            }, 'best_model.pth')

    # After training is complete, evaluate on test set
    print("\nEvaluating final model on test set:")
    test_accuracy = test(model, device, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main() 