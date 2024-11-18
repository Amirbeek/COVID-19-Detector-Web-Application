from flask import Flask, render_template, redirect, url_for
import os
import tensorflow as tf


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__, static_folder='templates/static')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model = tf.keras.models.load_model('Covid_X_Ray_Predictor.h5')  # Update this path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_covid')
def check_covid():
    render_template('covid.html')


if __name__ == '__main__':
    app.run(debug=True)
