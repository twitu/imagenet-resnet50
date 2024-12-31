import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model import ResNet50


class ImageNetPredictor:
    def __init__(self, checkpoint_path):
        # Initialize model using your wrapper
        self.model = ResNet50()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load state dict based on checkpoint structure
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Handle DataParallel prefix if present
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)  # Move model to device
        self.model.eval()

        # Load class labels
        with open("imagenet_classes.txt", "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Define image transforms - matching training preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        print("Model loaded successfully!")

    def predict(self, image):
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess image
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Get predictions
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = F.softmax(output[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)

        # Format results
        results = {}
        for prob, idx in zip(top5_prob, top5_indices):
            label = self.labels[idx]
            results[label] = float(prob)

        return results


# Initialize predictor
predictor = ImageNetPredictor("best_model.pt")

# Define Gradio interface
title = "ResNet50 ImageNet1k Classifier"
description = """
## ResNet50 Image Classification Model
This model classifies images into 1000 ImageNet categories using a custom ResNet50 architecture.

### Usage Tips:
- Upload or drag-and-drop an image
- The model works best with clear, well-lit images
- The main subject should be centered in the frame
- Supports common image formats (JPEG, PNG, etc.)

The model will return the top 5 predicted classes with their confidence scores.
"""

# Create interface
iface = gr.Interface(
    fn=predictor.predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title=title,
    description=description,
    allow_flagging="never",
)

# Launch app
if __name__ == "__main__":
    iface.launch()
