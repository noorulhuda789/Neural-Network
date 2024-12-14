import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from models.MultiModel import predict, LModelForward
from DataStore.loadingMultiData import load_parameters, loadClasses

# True labels for the test images
true_labels = ["Monitor", "Laptop", "Dog", "Laptop"]

# File paths for the test images
file_paths = [
    r"C:\Users\hp\OneDrive\Documents\AI\laptopMonitorReco\testPic\44.jpeg",
    r"C:\Users\hp\OneDrive\Documents\AI\laptopMonitorReco\testPic\255.jpg",
    r"C:\Users\hp\OneDrive\Documents\AI\laptopMonitorReco\testPic\fox.jpeg",
    r"C:\Users\hp\OneDrive\Documents\AI\laptopMonitorReco\testPic\tv.jpeg"
]

# Load model parameters and class names
parameters = load_parameters()
class_names = loadClasses()

# Function to predict the class of an image
def predict_image(file_path, parameters, class_names, px=64):
    img = Image.open(file_path).resize((px, px))
    my_image = np.array(img) / 255.0
    images = np.array([my_image])
    images = images.reshape(len(images), -1).T
    
    predicted_probs = predict(images, parameters, return_probs=True)
    predicted_class = np.argmax(predicted_probs)
    return class_names[predicted_class], predicted_probs

# Predict labels and probabilities for each test image
predicted_labels = []
probability_values = []
for file_path in file_paths:
    pred_label, pred_probs = predict_image(file_path, parameters, class_names)
    predicted_labels.append(pred_label)
    probability_values.append(pred_probs)

# Initialize a confusion matrix and a matrix to hold the max probabilities
cm = np.zeros((len(class_names), len(class_names)))
prob_cm = np.zeros((len(class_names), len(class_names)))

# Fill in the confusion matrix and probability matrix
for true, pred, probs in zip(true_labels, predicted_labels, probability_values):
    true_idx = class_names.index(true)
    pred_idx = class_names.index(pred)
    cm[true_idx, pred_idx] += 1
    prob_cm[true_idx, pred_idx] = max(prob_cm[true_idx, pred_idx], np.max(probs))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
