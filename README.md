# MNIST Java Microservice

A RESTful microservice that integrates a pre-trained TensorFlow ML model for MNIST digit classification, achieving almost 99% accuracy. Built with Spring Boot and TensorFlow Java.

## Features

- REST API for MNIST digit prediction
- Pre-trained TensorFlow model integration
- Docker containerized for easy deployment
- High accuracy classification

## Quick Start

### Run with Docker

```bash
docker run -p 8080:8080 shakhade/mnist-microservice
```
or via DockerHub

```bash
docker pull shakhade/mnist-microservice
```

### API Usage

Send a POST request to `http://localhost:8080/predict` with a JSON payload containing a 28x28 array of floats representing the image.

Example using Python:

```python
import requests
import json

# Example input (28x28 array)
data = {"image": [[0.0] * 28] * 28}  # Replace with actual image data
response = requests.post("http://localhost:8080/predict", json=data)
print(response.json())
```

## Build Locally

```bash
mvn clean package
docker build -t mnist-microservice .
```

## Technologies

- Java 17
- Spring Boot 3.2.5
- TensorFlow Java 0.5.0
- Docker

## Python Trained Model via Simple Neural Network (Train your own)

```python
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  

# Simple feedforward neural network
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
model.save('saved_model')
```
### Google Colab Export
If you are training your model on Google Colab's T4 GPU, you can use this snippet to export your model as ZIP file
```python
from google.colab import files
import shutil

# Zip the directory
shutil.make_archive("saved_model", 'zip', "saved_model")

# Download the zip file
files.download("saved_model.zip")
```