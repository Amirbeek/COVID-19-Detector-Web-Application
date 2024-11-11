import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('./covid_detection_model.h5')

def preprocess_image(image):
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = preprocess_image(request.FILES['image'])
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)
            label = np.argmax(prediction, axis=1)[0]
            labels = ['COVID', 'Normal', 'Virus']
            return JsonResponse({'prediction': labels[label]})
    else:
        form = ImageUploadForm()
    return render(request, 'myapp/upload.html', {'form': form})
