# test_model.py
from PIL import Image
import torch
from model_loader import model, device, CLASS_LABELS
from custom_preprocess import preprocess_image

image = Image.open("CC.jpg").convert("RGB")
tensor = preprocess_image(image)
outputs = model(tensor)

predicted_idx = outputs.argmax().item()  # Updated line
predicted_label = CLASS_LABELS[predicted_idx]
# Confidence score
probabilities = torch.nn.functional.softmax(outputs, dim=-1)  # Updated line
confidence = round(probabilities[0][predicted_idx].item() * 100, 2)
print(predicted_idx, predicted_label, probabilities, confidence)