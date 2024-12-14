import numpy as np
from PIL import Image
from DataStore.loadingMultiData import *
import matplotlib.pyplot as plt
def initializeParameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def linearForward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache

def linearActivationForward(A_prev, W, b, activation):
    if activation == "softmax":
        Z, linear_cache = linearForward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        Z, linear_cache = linearForward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def LModelForward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linearActivationForward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)
    AL, cache = linearActivationForward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)
    return AL, caches

def computeCost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(AL + 1e-8)) / m
    cost = np.squeeze(cost)
    return cost

def linearBackward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linearActivationBackward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linearBackward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linearBackward(dZ, linear_cache)
    return dA_prev, dW, db

def LModelBackward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    
    dAL = AL - Y
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linearActivationBackward(dAL, current_cache, activation="softmax")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linearActivationBackward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads


def updateParameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = Z_exp / np.sum(Z_exp, axis=0, keepdims=True)
    cache = Z
    return A, cache
    # e_x  =np.exp(Z - np.max(Z))
    # return e_x / e_x.sum(axis=0)

def softmax_backward(dA, cache):
    return dA
def predict_image(parameters, file_path, class_names, true_y=0, px=64):
    img = Image.open(file_path).resize((px, px))
    my_image = np.array(img) / 255.0
    images = np.array([my_image])
    images = images.reshape(len(images), -1).T
    
    predicted_probs = predict(images, parameters, return_probs=True)
    predicted_class = np.argmax(predicted_probs)
    confidence = predicted_probs[predicted_class]
    
    if confidence > 0.95:
        text = f"Predicted class: {class_names[predicted_class]}, Confidence: {confidence:.2f}"
        return text
    else:
        return "No predictions"


def predict(X, parameters, return_probs=False):
    AL, _ = LModelForward(X, parameters)
    if return_probs:
        return AL.squeeze()
    predictions = np.argmax(AL, axis=0)
    return predictions

def one_hot_encode(Y, num_classes):
    return np.eye(num_classes)[Y.reshape(-1)].T

# Main function to train the model
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=5000, print_cost=False):
    np.random.seed(1)
    costs = []

    # Initialize parameters
    parameters = initializeParameters(layers_dims)

    # One-hot encode Y
    Y = one_hot_encode(Y, layers_dims[-1])

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation
        AL, caches = LModelForward(X, parameters)

        # Compute cost
        cost = computeCost(AL, Y)

        # Backward propagation
        grads = LModelBackward(AL, Y, caches)

        # Update parameters
        parameters = updateParameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
        
            

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    save_parameters(parameters)

    

# Load data and initialize parameters
""" np.random.seed(1)
train_x, train_y = loadData()


print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")


dimensions = [12288, 180, 60, 25, 3]
 """

#parameters = L_layer_model(train_x, train_y, dimensions, num_iterations=2500, print_cost=True)
#parameters=load_parameters()


#predict_image(parameters,r"C:\Users\hp\OneDrive\Documents\AI\laptopMonitorReco\testPic\44.jpeg")
