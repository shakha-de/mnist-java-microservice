import random
import json
import requests
from tensorflow import keras
import matplotlib.pyplot as plt

# install tensorflow and matplotlib before
# requirements.txt was created via 'uv pip freeze > requirements.txt'
# uv pip install tensorflow matplotlib

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Randomly choose one image from the test set
random_index = random.randint(0, len(x_test) - 1)
random_image = x_test[random_index]
random_label = y_test[random_index]

# Normalize to float 0-1 (MNIST images are 0-255)
normalized_image = random_image.astype('float32') / 255.0

# Reshape to [1][28][28] for the endpoint (though the controller expects [28][28], but JSON sends as 3D)
# Actually, the endpoint takes float[][][], which in JSON is [[[floats]]], so [1][28][28]
json_body = json.dumps([normalized_image.tolist()])

# Display the image and its label
plt.imshow(random_image, cmap='gray')
plt.title(f"Label: {random_label}")
plt.axis('off')
plt.show()

# print(json_body)

# For local testing
url = "http://localhost:8080/api/mnist/predict"
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json_body, headers=headers)


print("Status Code:", response.status_code)
print("Response:", response.json())