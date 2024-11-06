import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

for dirpath, dirname, filenames in os.walk('COVID_IEEE'):
    print(f'there are {len(filenames)} files in {dirpath}')


data_dir = pathlib.Path('COVID_IEEE')
class_name = np.array([sorted(item.name for item in data_dir.glob("*"))])


def view_image(target, target_class):
    target_folder = os.path.join(target, target_class)
    random_image = random.sample(os.listdir(target_folder), 1)[0]
    print(random_image)  # Print the name of the random image
    img = mpimg.imread(os.path.join(target_folder, random_image))  # Correctly join the path to read the image
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    plt.show()
    return img

img =  view_image(data_dir, 'virus')
img =  view_image(data_dir, 'normal')

data = []
labels = []

covid = os.listdir(f"{data_dir}/covid")

for i in covid:
    image_path = f'{data_dir}/covid/{i}'  # Create the full path for the image
    image = cv2.imread(image_path)

    if image is not None:  # Check if the image was loaded successfully
        image = cv2.resize(image, (224, 224))  # Resize only if the image was loaded
        data.append(image)
        labels.append(0)
    else:
        print(f"Warning: Could not read image {image_path}")

normal = os.listdir(f"{data_dir}/normal")

for i in normal:
    image_path = f'{data_dir}/normal/{i}'  # Create the full path for the image
    image = cv2.imread(image_path)

    if image is not None:  # Check if the image was loaded successfully
        image = cv2.resize(image, (224, 224))  # Resize only if the image was loaded
        data.append(image)
        labels.append(1)
    else:
        print(f"Warning: Could not read image {image_path}")

virus = os.listdir(f"{data_dir}/virus")

for i in virus:
    image_path = f'{data_dir}/virus/{i}'  # Create the full path for the image
    image = cv2.imread(image_path)

    if image is not None:  # Check if the image was loaded successfully
        image = cv2.resize(image, (224, 224))  # Resize only if the image was loaded
        data.append(image)
        labels.append(2)
    else:
        print(f"Warning: Could not read image {image_path}")

img_data = np.array(data) / 255.0 # this is for normalize the data , i need to devide for 255
img_labels = np.array(labels)



data = np.array(data) / 255.0
img_labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, img_labels, test_size = 0.2, random_state = 42)
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_test = tf.keras.utils.to_categorical(y_train, num_classes = 10)


model = Sequential()

#Block Number 1
model.add(Conv2D(input_shape = (224,224,3), filters=32,padding="same", kernel_size= (3,3)))
model.add(Activation("relu"))

model.add(Conv2D(filters=32,padding="same", kernel_size= (3,3)))
model.add(Activation("relu"))

model.add(MaxPool2D((2,2)))

#Block Number 2
model.add(Conv2D(filters=64,padding="same", kernel_size= (3,3)))
model.add(Activation("relu"))


model.add(Conv2D(filters=64,padding="same", kernel_size= (3,3)))
model.add(Activation("relu"))

model.add(MaxPool2D((2,2)))

#Block Number 3
model.add(Conv2D(filters=128,padding="same", kernel_size= (3,3)))
model.add(Activation("relu"))

model.add(Conv2D(filters=128,padding="same", kernel_size= (3,3)))
model.add(Activation("relu"))

model.add(MaxPool2D((2,2)))

model.add(MaxPool2D((2,2)))

# Fully Connected layer
model.add(Flatten())

model.add(Dense(units=1024, activation="relu"))

model.add(Dense(units=256, activation="relu"))


model.add(Dense(units=3, activation="softmax"))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=15, batch_size=32)

# saving the model history
loss = pd.DataFrame(model.history.history)

# plotting the loss and accuracy
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(loss["loss"], label="Loss")
plt.plot(loss["val_loss"], label="Validation_loss")
plt.legend()
plt.title("Training and Validation Loss")

plt.subplot(2, 2, 2)
plt.plot(loss['accuracy'], label="Training Accuracy")
plt.plot(loss['val_accuracy'], label="Validation_ Accuracy ")
plt.legend()
plt.title("Training-Validation Accuracy")

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis = 1)
y_test_new = np.argmax(y_test, axis = 1)

print(classification_report(y_test_new, y_pred))

pd.DataFrame(confusion_matrix(y_test_new, y_pred),
             columns= ["covid", "normal", "virus"], index = ["covid", "normal", "virus"])


base_model = tf.keras.applications.MobileNet(input_shape=[224,224,3], weights = "imagenet", include_top=False)

for layer in base_model.layers:
  layer.trainable =False

  model = Flatten()(base_model.output)
  model = Dense(units=1024, activation="rule")(model)
  model = Dense(units=512, activation="rule")(model)
  model = Dense(units=256, activation="rule")(model)

  predictions_layer = Dense(units=3, activation="softmax")(model)


model = Model(inputs = base_model.input, outputs = predictions_layer)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


model.fit(X_train, y_train, validation_split=0.3, epochs=15, batch_size=32)


#saving the model history
loss = pd.DataFrame(model.history.history)

#plotting the loss and accuracy
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.plot(loss["loss"], label ="Loss")
plt.plot(loss["val_loss"], label = "Validation_loss")
plt.legend()
plt.title("Training and Validation Loss")

plt.subplot(2,2,2)
plt.plot(loss['accuracy'],label = "Training Accuracy")
plt.plot(loss['val_accuracy'], label ="Validation_ Accuracy ")
plt.legend()
plt.title("Training-Validation Accuracy")

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis = 1)
y_test_new = np.argmax(y_test, axis = 1)

pd.DataFrame(confusion_matrix(y_test_new, y_pred), columns= ["covid", "normal", "virus"], index = ["covid", "normal", "virus"])

