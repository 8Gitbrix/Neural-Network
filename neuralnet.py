# Ashwin Jeyaseelan May 2017
import numpy as np

class NeuralNetwork:
	# initialize the network-
	def __init__(self, units_per_layer_list, iteration, learn_rate):
    	#we also need to figure out the bias....
		self.weights = []
		self.bias = []
		self.z = [] # weights * inputs + bias for units in each layer
		self.a = [] # activation function applied to z for units in each layer
		self.alpha = learn_rate
		self.iteration = iteration
		l = len(units_per_layer_list)
		for i in range(l-1):
			self.weights.append(np.random.rand(units_per_layer_list[i+1],units_per_layer_list[i]))
		for i in range(1,l): # We want a bias for each unit except the input layer!
			self.bias.append(np.random.rand(units_per_layer_list[i]))

	def show_weights(self):
		print(self.weights)

	def dtanh(self,x):
		return 1.0 - np.tanh(x) ** 2

	def forward(self, input, label):
		if len(input) != self.weights[0].shape[1]:
			raise Exception('Invalid input size!')
		output = 0
		for w,b in zip(self.weights, self.bias):
			z = np.dot(input,w.T) + b
			self.z.append(z)
			input = np.tanh(z)
			self.a.append(input)
			output = input
		self.error = 0.5 * np.power(output-label,2)
		self.derror = output - label
		return output

	def backpropagate(self):
		delta = self.derror * self.dtanh(self.z[-1])
		self.weights[-1] -= self.alpha * self.a[-1] * delta
		self.bias[-1] -= self.alpha * delta
		for (i,a),z in zip(reversed(list(enumerate(self.a[:-1]))),reversed(self.z[:-1])):
			delta = np.dot(self.weights[i+1].T,delta) * self.dtanh(z)
			self.weights[i] -= self.alpha * np.dot(delta,a)
			self.bias[i] -= self.alpha * delta

	def train(self, input_data, input_labels):
		n = len(input_data)
		for i in range(self.iteration):
			average_error = 0
			for d,l in zip(input_data,input_labels):
				self.forward(d,l)
				average_error = average_error + self.error
				self.backpropagate()
				self.a = []
				self.z = []
			print("iteration #{} Error: {}" .format(i,average_error/n))

	def predict(self,input_data):
		print(self.forward(input_data,0))
		self.error = 0
		self.a = []
		self.z = []

m = NeuralNetwork([2,3,2,1], 6, 0.5)
m.train([[1,2]],[0.5])
