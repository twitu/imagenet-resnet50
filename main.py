from model import ImageNetModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from tqdm import tqdm
import torch
from train import DataLoader_ImageNet
from torch.cuda.amp import autocast, GradScaler
import os
import boto3
from pathlib import Path


def save_checkpoint(
    s3_client,
    bucket_name,
    model,
    optimizer,
    scheduler,
    epoch,
    loss,
    accuracy,
    checkpoint_dir,
    is_best=False,
):
    # Save regular checkpoint
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint_data, checkpoint_path)
    s3_client.upload_file(
        checkpoint_path, bucket_name, f"checkpoints/checkpoint_epoch_{epoch}.pt"
    )

    # If this is the best model so far, save it separately
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint_data, best_model_path)
        s3_client.upload_file(best_model_path, bucket_name, "checkpoints/best_model.pt")
        print(f"Saved best model with accuracy: {accuracy:.2f}%")


def get_latest_checkpoint(s3_client, bucket_name, checkpoint_dir):
    try:
        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="checkpoints/")
        if "Contents" not in objects:
            return None

        checkpoints = [
            obj["Key"] for obj in objects["Contents"] if obj["Key"].endswith(".pt")
        ]
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        local_path = f"{checkpoint_dir}/{os.path.basename(latest)}"

        # Download the checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        s3_client.download_file(bucket_name, latest, local_path)
        return local_path
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def train(model, device, train_loader, optimizer, scheduler, epoch, scaler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
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
        avg_loss = running_loss / len(train_loader)
        pbar.set_postfix({"loss": avg_loss})

    # Step scheduler with average loss for the epoch
    scheduler.step(avg_loss)
    return avg_loss


def validate(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Loss: {loss/len(test_loader):.3f}, Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    # Load hyperparameters from environment variables with defaults
    EPOCHS = int(os.environ.get("TRAINING_EPOCHS", 100))
    LR = float(os.environ.get("LEARNING_RATE", 0.01))
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.05))
    DATA_PATH = os.environ.get("DATA_PATH", "/mnt/ebs_volume/data")
    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME environment variable must be set")

    # Scheduler parameters
    SCHEDULER_FACTOR = float(os.environ.get("SCHEDULER_FACTOR", 0.1))
    SCHEDULER_PATIENCE = int(os.environ.get("SCHEDULER_PATIENCE", 10))
    SCHEDULER_THRESHOLD = float(os.environ.get("SCHEDULER_THRESHOLD", 0.0001))
    SCHEDULER_MIN_LR = float(os.environ.get("SCHEDULER_MIN_LR", 0))
    SCHEDULER_EPS = float(os.environ.get("SCHEDULER_EPS", 1e-08))

    # Print configuration
    print("\nTraining Configuration:")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LR}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    print(f"Scheduler Factor: {SCHEDULER_FACTOR}")
    print(f"Scheduler Patience: {SCHEDULER_PATIENCE}")
    print(f"Scheduler Threshold: {SCHEDULER_THRESHOLD}")
    print(f"Scheduler Min LR: {SCHEDULER_MIN_LR}")
    print(f"Scheduler Eps: {SCHEDULER_EPS}\n")

    # Initialize model
    model_wrapper = ImageNetModel()
    model = model_wrapper.model
    device = model_wrapper.device

    # Print model summary
    model_wrapper.get_model_summary()

    # Initialize data loader with verification
    #data_loader = DataLoader_ImageNet()
    #train_loader, val_loader, test_loader = data_loader.load_data(DATA_PATH)

    # Verify data loaders
    #try:
        #print(f"Number of test batches: {len(test_loader)}")
        #sample_batch, sample_labels = next(iter(train_loader))
        #print(f"Successfully loaded batch of shape: {sample_batch.shape}")
        #print(f"Number of training batches: {len(train_loader)}")
        #print(f"Number of validation batches: {len(val_loader)}")
        #sample_batch = sample_batch.to(device)
        #print(f"Data successfully moved to device: {sample_batch.device}")
    #except Exception as e:
        #print(f"Error in data loading verification: {str(e)}")
        #raise

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        verbose=True,
        threshold=SCHEDULER_THRESHOLD,
        threshold_mode="rel",
        cooldown=0,
        min_lr=SCHEDULER_MIN_LR,
        eps=SCHEDULER_EPS,
    )

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()

    # Create checkpoint directory
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)

    # Load latest checkpoint if it exists
    start_epoch = 0
    s3_client = boto3.client("s3")

    checkpoint_path = get_latest_checkpoint(s3_client, BUCKET_NAME, checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    best_acc = 0
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEPOCH: {epoch+1}/{EPOCHS}")
        train_loss = 0
        #train(
        #    model, device, train_loader, optimizer, scheduler, epoch, scaler
        #)

        #accuracy = validate(model, device, val_loader)
        # if accuracy > best_acc:
        #     best_acc = accuracy
        #     save_checkpoint(
        #         s3_client=s3_client,
        #         bucket_name=BUCKET_NAME,
        #         model=model,
        #         optimizer=optimizer,
        #         scheduler=scheduler,
        #         epoch=epoch,
        #         loss=train_loss,
        #         accuracy=accuracy,
        #         checkpoint_dir=checkpoint_dir,
        #         is_best=True,
        #     )

        # Save checkpoint every epoch
        save_checkpoint(
            s3_client,
            BUCKET_NAME,
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            #accuracy,
            checkpoint_dir,
        )

    # After training is complete, evaluate on test set
    print("\nEvaluating final model on test set:")
    #test_accuracy = validate(model, device, test_loader)
   # print(f"Final Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
