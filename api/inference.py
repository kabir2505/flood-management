# Model loading & inference functions

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

#{'no_damage': 0, 'visible_damage': 1}

def load_model():
    model = models.resnet18(pretrained=False)

    from torch import nn
    model.fc = nn.Linear(model.fc.in_features, 1)


    model.load_state_dict(torch.load("models/resnet18_weights.pth", map_location=torch.device("cpu")))


    model.eval()
    return model

def predict_image(model, image_path):
    # Define the preprocessing steps (same as used during training)
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Use ImageNet stats if pretrained; else custom
                             std=[0.229, 0.224, 0.225])
    ])

    # Open and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).item()
        binary_class = 1 if prediction >= 0.5 else 0
        print(prediction, binary_class)

    return binary_class, prediction  # e.g., (1, 0.89)
