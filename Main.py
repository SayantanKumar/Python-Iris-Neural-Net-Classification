import numpy as np #Used for managing the matrices
from sklearn import datasets as ds #Used to import the iris dataset
from scipy import optimize #Use the wrapper of the BFGS algorithm in scipy
from matplotlib import pyplot as plt #Plot graphs

np.seterr(all='ignore') #Because numbers can be very small, this hides overflow & /0 errors
iris = ds.load_iris() #Load the data set

#Put the data and labels into an x and y matrix
x = iris.data
y = iris.target

#Then normalize the data into a range between 0 & 1
xM = x.max()
x = x/x.max()
y = y/y.max()
y = np.reshape(y, (150,1)) #Reshaped the y because it was supposedly in the wrong shape


class NeuralNetwork(object):
	#Initialise the inital values for the NN
	def __init__(self):
		#Neural Network Model
		self.inputSize = 4 #4 Inputs, sepal length/width and petal length/width
		self.hiddenSize = 5 #Rounded mean of input & output, we'll see how well it works
		self.outputSize = 1 #1 Output to classify which flower it is
		
		#Create the weights randomly into a matrix of the same size as the number of nodes they are connected to 
		self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #input -> hidden
		self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #hidden -> output
		
	
	#Predict function, Use this after the network is trained to predict it by passing an array for the sizes and the number used to normalize the training data
	def predict(self, x):
		x = np.array(x)
		prediction = self.forwardProp((x/x.max())) * 2 #Forward propagates the normalized array of data, then de-normalizes the output
		if prediction < 0.5:
			print("Flower Prediction: Setosa\nNumeric Prediction: "+ str(prediction[0])) #Then prints out the name of the flower via comparitives, as well as the value for prediction
		elif prediction < 1.5:
			print("Flower Prediction: Versicolor\nNumeric Prediction: "+ str(prediction[0]))
		elif prediction < 2.5:
			print("Flower Prediction: Virginica\nNumeric Prediction: "+ str(prediction[0]))
	
	
	#Propagate the data forward through the network using sigmoid function as the activation function
	def forwardProp(self, x):
		self.z2 = np.dot(x, self.W1) #Z's are the dot product of the output from the previous nodes and the weights
		self.a2 = self.sigmoid(self.z2) #A and yHat are the z's but with the activation function applied
		self.z3 = np.dot(self.a2, self.W2)
		self.yHat = self.sigmoid(self.z3)
		return self.yHat
	
	
	#Sigmoid equation for use as the activation function
	def sigmoid(self, z):
		return 1/(1+np.exp(-z))
		
	
	#Cost function to work out how wrong we are when training - Used in gradient descent to reduce the cost
	#Error = 0.5(target-predicted)^2
	def costFunction(self, x, y):
		self.yHat = self.forwardProp(x)
		J = 0.5*sum((y-self.yHat)**2) #cost function to work out how wrong we were, the difference between the actual and predicted, squared then halved
		return J
	
	
	#Derived sigmoid function used in gradient descent as part of getting the overall gradient
	def sigmoidDerived(self, z):
		return ((np.exp(-z)) / ((1 + np.exp(-z))**2))
	

	#Derived cost function also used in gradient descent as part of getting the overall gradient
	#The function also works out how much to change the weights using the delta rule
	#Change in weight = (target output-predicted) * derived function * input
	def costFunctionDerived(self, X, y):
		self.yHat = self.forwardProp(X)   
		#Weight Layer 1
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidDerived(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)
		#Weight Layer 2
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidDerived(self.z2)
		dJdW1 = np.dot(X.T, delta2)  
		return dJdW1, dJdW2
	
	
	#Combines the 2 weights matrices into one 
	def getParams(self):
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params
	
	
	#Reset weights from the new single matrix back into 2 matrices
	def setParams(self, params):
		W1_start = 0
		W1_end = self.hiddenSize * self.inputSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputSize , self.hiddenSize))
		W2_end = W1_end + self.hiddenSize*self.outputSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenSize, self.outputSize))

		
	#Return the change in weights as one matrix
	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costFunctionDerived(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
	
	
	#Reset the weights matrices then add the current cost to the list of costs
	def callbackf(self, params):
		self.setParams(params) #Reset the weight matrices
		self.J.append(self.costFunction(self.X, self.Y)) #Add the cost of the current weights to the cost array
		
	
	#Resets the weight matrices then passes the current costs and change in weights
	def costFunctionWrapper(self, params, X, y):
		self.setParams(params)
		return self.costFunction(X, y), self.computeGradients(X, y)
	
	
	#The main train function that uses scipy's optimizing wrapper of the BFGS algorithm
	def train(self, X, y):
		#Create variables for local use
		self.X = X
		self.Y = y
	 
		self.J = [] #Create list to hold costs for graph
		params0 = self.getParams() #Get the weights in one matrix for optimization
		options = {'maxiter': 3500, 'disp' : False} #Set options for optimization - set disp to true to get more details when training
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackf) #And optimize
		
		self.setParams(_res.x) #Set the new weights from the outcome of the optimization

	
	#Seperate training function for a demo which uses 1 iter at a time
	def train1(self, X, y):

		self.X = X
		self.Y = y
		self.J = []
		params0 = self.getParams()
		options = {'maxiter': 1, 'disp' : False}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackf)
		self.setParams(_res.x)




#-----------------------------------------------Demos -----------------------------------------------#
#---Run the file for the main demo or import it to run the other demos or your own code on the net---#

#Main demo, includes example data - runs when file is ran alone
def demo():
	net = NeuralNetwork()
	costBefore = float(net.costFunction(x,y)[0])
	net.train(x, y)
	costAfter = float(net.costFunction(x,y)[0])

	print("Cost Before: " + str(costBefore))
	print("Cost After: " + str(costAfter))
	print("Cost difference: " + str(costBefore - costAfter))

	net.predict([6.7,5.2,2.5,1.6])

	plt.plot(net.J)
	plt.grid(1)
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.show()


#Gets the flower based on inputted values
def inp():
	net = NeuralNetwork()
	net.train(x,y)
	sepalL = float(input("Sepal Length: "))
	sepalW = float(input("Sepal Width : "))
	PetalL = float(input("Petal Length: "))
	PetalW = float(input("Petal Width : "))
	inps = [sepalL,sepalW,PetalL,PetalW]
	net.predict(inps)

#Shows interactive graph as net is being trained
def training(_iters):
	js = np.array([])
	net = NeuralNetwork()
	plt.ion()
	for iters in range(_iters):
		net.train1(x, y)
		js= np.append(js, net.J)
		js.reshape(1, len(js))
		plt.plot(js)
		plt.grid(1)
		plt.xlabel('Iterations')
		plt.ylabel('Cost')
		plt.show()
		plt.pause(0.01)
	while True: plt.pause(0.01)

if __name__ == "__main__":
	demo()