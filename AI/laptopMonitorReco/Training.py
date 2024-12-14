import numpy as np
import matplotlib.pyplot as plt
from models.lmodel import *
from DataStore.loadData import *
from DataStore.memmap import *


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
    