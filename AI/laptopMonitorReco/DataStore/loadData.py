import os
import pickle
from PIL import Image
import numpy as np
from models.lmodel  import *

def load_data(px=64):
    images = []
    labels = []
    for label in ["1", "0"]:
        folder = label
        # folder = os.path.join("/kaggle/input/catsndogs", label)
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).resize((px, px))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(1 if label == "1" else 0)
    images = np.array(images).reshape(len(images), -1).T
    labels = np.array(labels).reshape(1, len(labels))
    return images, labels

def save_parameters(parameters):
    parameters_file = open("Parameters", "wb")
    pickle.dump(parameters, parameters_file)
    parameters_file.close()


def load_parameters():
    parameters_file = open("Parameters", "rb")
    parameters = pickle.load(parameters_file)
    parameters_file.close()
    return parameters


        
def calculate_accuracy(predictions, true_labels):
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    accuracy = np.mean(predictions == true_labels) * 100
    return accuracy