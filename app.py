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

app = Flask(__name__, static_folder='templates/static')
app.config['SECRET_KEY'] = 'covid19predictor'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/covid', methods=['GET', 'POST'])
def covid():
    form = IF()
    message = ''
    result = None
    virus = None
    if form.validate_on_submit():
        file = form.image.data
        file_bytes = file.read()

        # Load the model and class indices
        interpreter = tf.lite.Interpreter(model_path='./models/Covid_X_Ray_Predictor2.tflite')
        interpreter.allocate_tensors()
        class_indices = joblib.load('class_indices.joblib')  # Load updated class indices
        print(f"Loaded class indices: {class_indices}")

        # Decode and preprocess the image
        covid_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        covid_image = cv2.resize(covid_image, (224, 224)) / 255.0
        covid_image = np.expand_dims(covid_image, axis=0).astype(np.float32)

        # Perform prediction
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        interpreter.set_tensor(input_details['index'], covid_image)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details['index'])
        predicted_label = np.argmax(prediction)
        print(f"Prediction: {prediction}")
        print(f"Predicted Label: {predicted_label}")
        virus = predicted_label
        #nt(virus)

        matching_classes = [key for key, value in class_indices.items() if value == predicted_label]
        if matching_classes:
            predicted_class = matching_classes[0]
            message = f"Predicted Class: {predicted_class}"
            result = f"Prediction: {predicted_class}"
        elif int(predicted_label) == 2:
                message = f"Predicted Class: (Virus)"
                result = "Virus detected"
        else:
            message = "Prediction failed: No matching class found"
            result = "No matching result"

    return render_template('covid.html', form=form, message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
