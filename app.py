import time
import joblib
from flask import Flask, render_template, request
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import cv2
from PIL import Image
import numpy as np
from ImageForm import ImageForm as IF
from tflite_predictor import load_model  # Ensure load_model handles tflite model loading

app = Flask(__name__, static_folder='templates/static')
app.config['SECRET_KEY'] = 'covid19predictor'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/covid')
def covid():
    form = IF()
    return render_template('covid.html', form=form)


@app.route('/covid', methods=['GET', 'POST'])
def covid_checker():
    form = IF()  # Create the form instance
    message = ''
    result = None

    if form.validate_on_submit():  # Check if the form is valid
        file = form.image.data

        # Load the TensorFlow Lite model and class indices
        interpreter = load_model('./models/Covid_X_Ray_Predictor.tflite')
        class_indices = joblib.load('class_indices.joblib')

        # Read the file as bytes
        file_bytes = file.read()

        # Decode the image using OpenCV
        covid_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        covid_image = cv2.resize(covid_image, (224, 224)) / 255.0
        covid_image = np.expand_dims(covid_image, axis=0).astype(np.float32)

        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], covid_image)
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(prediction)

        predicted_class = [key for key, value in class_indices.items() if value == predicted_label][0]

        return f"Predicted Class: {predicted_class}"

    return render_template('covid.html', form=form, message=message, result=result)


if __name__ == '__main__':
    app.run(debug=True)
