from django.conf import settings
import torch
from transformers import ViTForImageClassification, ViTConfig
import os
import torch
import timm
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def load_custom_model():
    # Initialize the model from pretrained weights
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)

    base_path= os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'models/custom_vit_model.pth')

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)


    model.eval()  # Set to evaluation mode
    return model.to(device)

# Load class labels from file
CLASS_LABELS = ['Benign', 'Malignant']

# Initialize device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_custom_model()
