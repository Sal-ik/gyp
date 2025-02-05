from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image
import torch
from .model_loader import model, device, CLASS_LABELS
from .custom_preprocess import preprocess_image

def home(request):
    return render(request, 'upload.html')

def classify(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        image_path = fs.path(filename)
        
        try:
            # Open the uploaded image
            image = Image.open(image_path).convert("RGB")
            # Custom preprocessing
            image_tensor = preprocess_image(image)
    
            outputs = model(image_tensor)
    
            # Get prediction
            predicted_idx = outputs.argmax().item()  # Updated line
            predicted_label = CLASS_LABELS[predicted_idx]
            
            # Confidence score
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)  # Updated line
            confidence = round(probabilities[0][predicted_idx].item() * 100, 2)
            
            return render(request, 'result.html', {
                'image_url': fs.url(filename),
                'prediction': predicted_label,
                'confidence': confidence
            })
            
        except Exception as e:
            return render(request, 'error.html', {'error': str(e)})
        
        finally:
            pass
    
    #return redirect('home')
