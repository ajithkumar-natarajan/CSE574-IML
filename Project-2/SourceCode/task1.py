"""
Created on 10/17/19 @ 16:54:10

@author: ajithkumar-natarajan
"""

import numpy as np
import matplotlib as plot

def load_mnist(path, kind='train'):
	import os
	import gzip
	import numpy as np

	"""Load MNIST data from `path`"""
	labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
	images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

	return images, labels


#Function to normalize feature values. Moves the point to origin and divides by range of the corresponding column.
def normalize(data):
	normalized_data = data.copy()
	for column in data.columns:
		max_value = data[column].max()
		min_value = data[column].min()
		normalized_data[column] = (data[column] - min_value) / (max_value - min_value)
	return normalized_data


#Sigmoid function definition which is the activation function.
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def relu(z):
	if(z < 0):
		return 0
	else:
		return z

x_train, y_train = load_mnist('/home/ajithkumar/Documents/MS_UB/Coursework/Fall2019/CSE574_ML/Projects/Projects-2/SourceCode/data_with_notebook/data/fashion', kind='train')
x_test, y_test = load_mnist('/home/ajithkumar/Documents/MS_UB/Coursework/Fall2019/CSE574_ML/Projects/Projects-2/SourceCode/data_with_notebook/data/fashion', kind='t10k')

w1 = np.random.randn(x_train.shape[1], number_of_hidden_nodes)*0.1
b = 0

training_loss = list()
validation_loss = list()


# print(X_train)
# print(y_train)
print(x_train.shape)
print(w1.shape)
print(y_train.shape)
