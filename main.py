from model import ImageNetModel
from train import Trainer

def main():
    # Initialize model
    model = ImageNetModel()
    
    # Print model summary
    model.get_model_summary()
    
    # Initialize trainer
    trainer = Trainer(model.model)
    
    # Load data
    data_path = 'path/to/imagenet/dataset'  # Replace with your ImageNet dataset path
    trainer.load_data(data_path)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 