import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from  DataStore.loadData import *
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert(dZ.shape == Z.shape)
    return dZ



def initializeParameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def linearForward(A, W, b):
    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linearActivationForward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linearForward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linearForward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
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
    AL, cache = linearActivationForward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches

def computeCost(AL, Y):
    m = Y.shape[1]
    epsilon = 1e-8
    AL = np.clip(AL, epsilon, 1 - epsilon)
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

def linearBackward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    return dA_prev, dW, db

def linearActivationBackward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linearBackward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linearBackward(dZ, linear_cache)
    return dA_prev, dW, db

def LModelBackward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    epsilon = 1e-8
    AL = np.clip(AL, epsilon, 1 - epsilon)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linearActivationBackward(dAL, current_cache, activation="sigmoid")
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

def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1,m))
    
    
    probas, caches = LModelForward(X, parameters)

    
    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
def predict_image(parameters, file_path, true_y=0, px=64):
        img = Image.open(file_path).resize((px, px))
        image = np.array(img)
        my_image = np.array(img) / 255.0
        my_image = image.reshape(px * px * 3, 1)

        my_label_y = true_y

        my_predicted_image = predict(my_image, my_label_y, parameters)
        pred_test = calculate_accuracy(my_predicted_image,my_label_y)
        print(pred_test)
        text = "y = " + str(np.squeeze(my_predicted_image)) + ", model predicts a "
       
        if my_predicted_image == 1:
            text += "clean"
        else:
            text += "messy"

        return text
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 5000, print_cost=False):
    np.random.seed(1)
    costs = []
    loss=[] 
    parameters = initializeParameters(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = LModelForward(X, parameters)
        cost = computeCost(AL, Y)
        grads = LModelBackward(AL, Y, caches)
        parameters = updateParameters(parameters, grads, learning_rate)
       
        costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    #plt.plot(np.squeeze(costs))
    #plt.xlabel('iterations (per hundreds)')
    #plt.title("Learning rate =" + str(learning_rate))
    #plt.show()
    save_parameters(parameters)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('Loss')
    plt.xlabel('iterations ')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
train_x, train_y = load_data()
dimensions = [12288, 120, 60, 5, 1]
L_layer_model(train_x, train_y, dimensions, num_iterations=2500, print_cost=True) 