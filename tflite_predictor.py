from flask import Flask, render_template, request
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from ImageForm import ImageForm as IF

# Initialize Flask app
app = Flask(__name__, static_folder='templates/static')

# Load the TFLite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Perform inference on the model
def predict(interpreter, input_data):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess input_data if necessary (resize, normalize, etc.)
    input_data = np.array(input_data, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Flask configurations
app.config['SECRET_KEY'] = 'covid19predictor'
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Path to store uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size 16 MB

# Load the TFLite model
model_path = './models/Covid_X_Ray_Predictor.tflite'
interpreter = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/covid', methods=['GET', 'POST'])
def covid_checker():
    form = IF()  # ImageForm instance
    message = ''
    result = None

    if form.validate_on_submit():
        file = form.image.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = Image.open(file_path)
        image = image.resize((224, 224))  # Resize to the expected model input size (adjust as needed)
        input_data = np.array(image).astype(np.float32)

        # If the model expects normalization (e.g., pixel values between 0 and 1)
        input_data /= 255.0  # Normalize the input image

        # Perform prediction using the TFLite model
        prediction = predict(interpreter, input_data)

        # Interpret the prediction result
        result = 'Positive' if prediction > 0.5 else 'Negative'  # Example threshold for classification

    return render_template('covid.html', form=form, message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
