from django.shortcuts import render

# predictor/views.py
from django.shortcuts import render
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

MODEL = load_model("animal_classifier.h5")
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def predict_image(request):
    prediction = None
    img_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        img_path = os.path.join('media', img.name)
        
        with open(img_path, 'wb+') as f:
            for chunk in img.chunks():
                f.write(chunk)

        # Preprocess
        img_obj = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img_obj) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = MODEL.predict(img_array)
        prediction = CLASS_NAMES[np.argmax(pred)]
        img_url = '/' + img_path

    return render(request, 'predictor/index.html', {'prediction': prediction, 'image_url': img_url})

from django.template.loader import get_template
from django.http import HttpResponse

