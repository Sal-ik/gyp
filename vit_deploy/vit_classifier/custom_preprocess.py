from torchvision import transforms

# Example: Match preprocessing from your training
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    return custom_transform(image).unsqueeze(0)  # Add batch dimension