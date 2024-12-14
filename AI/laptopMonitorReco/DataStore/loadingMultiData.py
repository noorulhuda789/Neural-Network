import os
import pickle
from PIL import Image
import numpy as np

import os
import numpy as np
from PIL import Image

def loadData(px=64, folders=[]):
    images = []
    labels = []

    for index, folder in enumerate(folders):
        for filename in os.listdir(folder):
            if filename.endswith(".jpg",".png"):
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).resize((px, px))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(index)  # Use numeric label directly

    images = np.array(images)
    labels = np.array(labels)

    # Flatten images
    images = images.reshape(len(images), -1).T
    labels = labels.reshape(1, len(labels))

    return images, labels

# Usage

def save_parameters(parameters):
    parameters_file = open("Parameters", "wb")
    pickle.dump(parameters, parameters_file)
    parameters_file.close()

def saveClasses(classes):
    parameters_file = open("Classes", "wb")
    pickle.dump(classes, parameters_file)
    parameters_file.close()
    
def load_parameters():
    parameters_file = open("Parameters", "rb")
    parameters = pickle.load(parameters_file)
    parameters_file.close()
    return parameters

def loadClasses():
    parameters_file = open("Classes", "rb")
    parameters = pickle.load(parameters_file)
    parameters_file.close()
    return parameters
    
