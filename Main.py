

# In[55]:

import numpy as np
from sklearn import datasets as ds


# In[56]:

iris = ds.load_iris() #Load the data set

#Put the data and labels into an x and y matrix
x = iris.data #(150,4)
y = iris.target #(150,1)

#Then normalize the data into a range between 0 & 1
x = x/x.max()
y = y/y.max()


# In[57]:

class NeuralNetwork(object):
    def __init__(self):
        np.random.seed(1) #Sets a random seed, this will help when debugging as all the random weights will be the same every time it is run

        #Neural Network Model
        self.inputSize = 4 #4 Inputs, sepal length/width and petal length/width
        self.hiddenSize = 3 #Rounded mean of input & output, we'll see how well it works
        self.outputSize = 1 #1 Output to classify which flower it is

        #Create the weights randomly into a matrix of the same size as the number of nodes they are connected to
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #input -> hidden
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #hidden -> output

    def forwardProp(self, x):
        #Propagrate all the data forwards through the network using sigmoid as our activation function
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)
        return self.yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))


# In[58]:

net = NeuralNetwork()


# In[59]:

net.forwardProp(x)
