
# coding: utf-8

# In[6]:

import numpy as np
from sklearn import datasets as ds
from scipy import optimize
from matplotlib import pyplot as plt


# In[7]:

iris = ds.load_iris() #Load the data set

#Put the data and labels into an x and y matrix
x = iris.data #(150,4)
y = iris.target #(150,1)

#Then normalize the data into a range between 0 & 1
xM = x.max()
print(y.max())
x = x/x.max()
y = y/y.max()
y = np.reshape(y, (150,1))
y.shape


# In[38]:

class NeuralNetwork(object):
    def __init__(self):
        np.random.seed(1) #Sets a random seed, this will help when debugging as all the random weights will be the same every time it is run
        
        #Neural Network Model
        self.inputSize = 4 #4 Inputs, sepal length/width and petal length/width
        self.hiddenSize = 5 #Rounded mean of input & output, we'll see how well it works
        self.outputSize = 1 #1 Output to classify which flower it is
        
        #Create the weights randomly into a matrix of the same size as the number of nodes they are connected to 
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #input -> hidden
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #hidden -> output
        
    #Predict function, you will be able to use this after the network is trained to predict it by passing an array for the sizes and the number used to normalize the training data
    def predict(self, x, xM):
        prediction = self.forwardProp((x/xM)) * 2 #Forward propagates the normalized array of data, then de-normalizes the output
        if prediction < 0.5:
            return "Setosa", prediction #Then prints out the name of the flower via comparitives, as well as the value for prediction
        elif prediction < 1.5:
            return "Versicolor", prediction
        elif prediction < 2.5:
            return "Virginica", prediction
        else:
            return "ERROR", prediction #If for whatever reason the value is wayyyyy out
    
    
    
    
    def forwardProp(self, x):
        #Propagrate all the data forwards through the network using sigmoid as our activation function
        self.z2 = np.dot(x, self.W1) #Z's are the dot product of the output from the previous nodes and the weights
        self.a2 = self.sigmoid(self.z2) #A and yHat are the z's but with the activation function applied
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)
        return self.yHat
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z)) #Sigmoid equation, used for activation
        
    def costFunction(self, x, y):
        self.yHat = self.forwardProp(x)
        J = 0.5*sum((y-self.yHat)**2) #cost function to work out how wrong we were, the difference between the actual and predicted, squared then halved
        return J
    
    def sigmoidDerived(self, z):
        return ((np.exp(-z)) / ((1 + np.exp(-z))**2)) #Sigmoid but partially derived, this is used in gradient decent to alter the weights
    

    
    def costFunctionDerived(self, X, y):
        #The cost function but partially derived with respect to W and W2 for a given X and y, this is done 
        self.yHat = self.forwardProp(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidDerived(self.z3)) #The delta rule
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidDerived(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    
    
    def getParams(self):
        #Combines the 2 weights matrices into one 
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Reset weights from the new single matrix back into 2 matrices
        W1_start = 0
        W1_end = self.hiddenSize * self.inputSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputSize , self.hiddenSize))
        W2_end = W1_end + self.hiddenSize*self.outputSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenSize, self.outputSize))

        
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionDerived(X, y) #Work out the gradients for gradient decent
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel())) #Then return the 2 gradients as 1 matrix
    
    def callbackf(self, params):
        self.setParams(params) #Reset the weight matrices
        self.J.append(self.costFunction(self.X, self.Y)) #Add the cost of the current weights to the cost array
        
    def costFunctionWrapper(self, params, X, y):
        self.setParams(params) #Reset the weight matrices
        cost = self.costFunction(X, y) #Get the cost of the current weights
        grad = self.computeGradients(X,y) #Get the gradient of the current weights
        return cost, grad
    
    
    def train(self, X, y):
        #Create variables for local use
        self.X = X
        self.Y = y

        #Create list to hold costs
        self.J = []
        
        params0 = self.getParams() #Get the weights in one matrix for optimization

        options = {'maxiter': 3500, 'disp' : False} #Set options for optimization - set disp to true to get more details when training
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',                                  args=(X, y), options=options, callback=self.callbackf) #And optimize
        
        self.setParams(_res.x) #Set the new weights from the outcome of the optimization
        #self.optimizationResults = _res


# In[56]:

net = NeuralNetwork()


# In[60]:

costBefore = float(net.costFunction(x,y)[0])
net.train(x, y)
costAfter = float(net.costFunction(x,y)[0])

print("Cost Before: " + str(costBefore))
print("Cost After: " + str(costAfter))
print("Cost difference: " + str(costBefore - costAfter))

#Uncomment to show graph
#plt.plot(net.J)
#plt.grid(1)
#plt.xlabel('Iterations')
#plt.ylabel('Cost')
#plt.show()


# In[58]:

net.predict([6.7,5.2,2.5,1.6], xM) #Demo - some sizes in a 1:4 array, as well as the xM variable


# In[ ]:




# In[ ]:




# In[ ]:



