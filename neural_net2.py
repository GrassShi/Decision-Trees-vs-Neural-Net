#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:54:01 2021

@author: graceshi
"""

from joblib.numpy_pickle_utils import xrange
from numpy import *
import sklearn.metrics as metrics

  
class NeuralNet(object):
    def __init__(self):
        # Generate random numbers
        random.seed(1)
        
        # Assign random weights to a matrix,
        self.synaptic_weights = 2 * random.random((13,1)) - 1
  
    # The Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
  
    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
  
    # Train the neural network and adjust the weights each time.
    def train(self, inputs, outputs, training_iterations):
        for iteration in xrange(training_iterations):
            # Pass the training set through the network.
            output = self.learn(inputs)
  
            # Calculate the error
#            error = accuracy_score(outputs, output)*100
            error = outputs - output
  
            # Adjust the weights by a factor
            factor = dot(inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += factor
  
        # The neural network thinks.
  
    def learn(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
    
    
  
  
if __name__ == "__main__":
    # Initialize
    neural_network = NeuralNet()
  
#    # The training set.
#    inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
#    outputs = array([[1, 0, 1]]).T
#    print (inputs, outputs)
#    
#    neural_network.train(inputs, outputs, 10000)
#    
#    print(neural_network.learn([1,0,1]))
    
    # import data
    data =  pd.read_csv('heart.csv')
    X = data.values[:, 0: 13]
    Y = data.values[:, 13]
    

    
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
    
    X_train = array(X_train)
    Y_train = array([y_train]).T
    

    # Train the neural network
    neural_network.train(X_train, Y_train, 10000)
    
    print(neural_network.synaptic_weights)
  
    # Test the neural network with a test example
    #Mean absolute error regression loss
    print(metrics.mean_absolute_error(y_test,neural_network.learn(X_test))*100) 
#    y_pred = neural_network.learn(X_test)
#    error = subtract(y_test, y_pred.T)
#    print ((error))