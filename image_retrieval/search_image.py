import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
from PIL import Image
import pickle

# Function to creative model :
def extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs = vgg16_model.inputs , outputs = vgg16_model.get_layer("fc1").output )
    return extract_model

# function to preprocessing , convert img to tensor

def image_preprocessing(img):
    img = img.resize((224,224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# function to extract vector from image
def extract_vector(model, image_path):
    img = Image.open(image_path)
    img_tensor = image_preprocessing(img)

    vector = model.predict(img_tensor)[0]
    vector = vector / np.linalg.norm(vector)
    return vector

paths = pickle.load(open("paths.pkl","rb"))
vectors = pickle.load(open("vectors.pkl","rb"))
search_image_path = "test_image/lion.jpeg"

model = extract_model()
search_vector = extract_vector(model , search_image_path)
distance = np.linalg.norm(vectors - search_vector , axis = 1)
ids = np.argsort(distance)[:5]
anh = Image.open(paths[ids[1]])
plt.imshow(anh)
plt.show()