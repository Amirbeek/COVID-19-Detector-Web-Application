import imageio.v2 as imageio
from flask import Flask, request, render_template, flash, jsonify
import cv2
import tensorflow as tf
import joblib
import numpy as np
import requests
import csv
import os

FORMCARRY_URL = "https://formcarry.com/s/YOUR_FORMCARRY_ID"
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CSV_FILE = "feedback_data.csv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/covid', methods=['GET', 'POST'])
def covid():
    prediction = None  # Initialize to None
    if request.method == 'POST':
        image_file = request.files.get('images[]')
        if image_file:
            try:
                image = imageio.imread(image_file)
                processed_image = preprocess_image(image)
                prediction = predict(processed_image)
                print(prediction)
                return jsonify({"prediction": prediction})
            except Exception as e:
                app.logger.error(f"Failed to process or predict the image. Error: {str(e)}")
                flash(f"Failed to process or predict the image. Error: {str(e)}", 'error')
                prediction = None
        else:
            flash("No image uploaded", 'error')
            print("No image was uploaded")
    else:
        print("This was a GET request, so no prediction.")

    print("Final Prediction being sent to template:", prediction)
    return render_template('covid.html', prediction=prediction,)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        try:
            form_data = {
                "name": request.form.get("name"),
                "email": request.form.get("email"),
                "message": request.form.get("message")
            }
            response = requests.post(FORMCARRY_URL, data=form_data)
            if response.status_code == 200:
                return jsonify({"success": True, "message": "Feedback submitted successfully!"})
            else:
                return jsonify({"success": False, "message": "Failed to submit feedback. Try again later."}), 400
        except Exception as e:
            return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

    return render_template('feedback.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'POST':
        try:
            form_data = {
                "upload_method": request.form.get("upload_method"),
                "purpose": request.form.get("purpose"),
                "accuracy": request.form.get("accuracy"),
                "expectation": request.form.get("expectation"),
                "features": request.form.getlist("features"),
                "recommend": request.form.get("recommend"),
                "comments": request.form.get("comments")
            }
            save_feedback_to_csv(form_data)
            return jsonify({"success": True, "message": "Feedback submitted successfully!"})
        except Exception as e:
            return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500
    return render_template('Questionnaire.html')

def save_feedback_to_csv(data):
    """Save feedback data to a CSV file."""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Upload Method", "Purpose", "Accuracy", "Expectation", "Features", "Recommend", "Comments"])
        writer.writerow([data["upload_method"], data["purpose"], data["accuracy"], data["expectation"], ', '.join(data["features"]), data["recommend"], data["comments"]])

def preprocess_image(image):
    """Preprocess the image for models prediction"""
    covid_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    covid_image = cv2.resize(covid_image, (224, 224))
    covid_image = covid_image / 255.0
    covid_image = np.expand_dims(covid_image, axis=0).astype(np.float32)
    return covid_image

def predict(image):
    """Predict the class of the image using the preloaded model and return probabilities as percentages."""
    interpreter = tf.lite.Interpreter(model_path='./models/Covid_model.tflite')
    interpreter.allocate_tensors()
    class_indices = joblib.load('./models/labels.joblib')
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details['index'], image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    logits = interpreter.get_tensor(output_details['index'])
    probabilities = softmax(logits[0])
    predictions_percentages = probabilities * 100
    predicted_classes = {class_indices[i]: f"{prob:.2f}%" for i, prob in enumerate(predictions_percentages)}

    return predicted_classes

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    temperature = 0.5 + np.random.random() * 0.5
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()

if __name__ == '__main__':
    app.run(debug=True)