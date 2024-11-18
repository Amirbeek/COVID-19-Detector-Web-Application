import tkinter as tk
from tkinter import filedialog
from tkinter import Label
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("Covid_X_Ray_Predector.h5")
class_names = ["covid", "normal", "virus"]  # Update to match your actual class names


def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, target_size)  # Resize image to target size
        image = image / 255.0  # Normalize to match training scale
        return np.expand_dims(image, axis=0)  # Add batch dimension
    else:
        raise ValueError(f"Could not read image from path: {image_path}")


def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and display the image
        image = Image.open(file_path)
        image = image.resize((224, 224))
        img_tk = ImageTk.PhotoImage(image)

        # Update image display
        img_label.configure(image=img_tk)
        img_label.image = img_tk

        # Make prediction
        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}")


# Set up the GUI
root = tk.Tk()
root.title("COVID-19 Classification")
root.geometry("400x400")

# Image display label
img_label = Label(root)
img_label.pack()

# Prediction result label
result_label = Label(root, text="Upload an image to predict", font=("Arial", 14))
result_label.pack(pady=10)

# Upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

root.mainloop()
