"""
Created on 10/17/19 @ 16:54:10

@author: ajithkumar-natarajan
"""

#Importing necessary libraries
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools


#Function to load data
def load_mnist(path, kind='train'):
  import os
  import gzip

  """Load MNIST data from `path`"""
  labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte.gz'
#                                 % kind)
  images_path = os.path.join(path,
                                '%s-images-idx3-ubyte.gz'
#                                 % kind)
  with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                offset=8)

  with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 784)

  return images, labels
  
#Sigmoid function 
def sigmoid(z):
  a = 1/(1 + np.exp(-z))
  return a


#Sigmoid derivative 
def sigmoid_derivative(a):
  return a*(1 - a)


#Softmax function
def softmax(x):
  a = np.exp(x - np.max(x))
  return a / a.sum(axis = 1, keepdims = True)


#Loss function 
def compute_loss(a, y):
  return np.sum(-np.log(a[range(y.shape[0]), y]))/y.shape[0]


#Reading input data
x_train, y_train = load_mnist('data/fashion', kind='train')
x_test, y_test = load_mnist('data/fashion', kind='t10k')


#Normalizing input data
normalized_x_train = x_train/255
normalized_x_test = x_test/255


#Initializing values
x1 = normalized_x_train
x2 = normalized_x_test
y1 = y_train
y1_test = y_test
epochs = 1000
learning_rate = 0.01
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
y2 = y_train.reshape(len(y_train), 1)
y2_test = y_test.reshape(len(y_test), 1)
y2 = onehot_encoder.fit_transform(y2)
y2_test = onehot_encoder.fit_transform(y2_test)
m = y1.shape[0]
no_of_hidden_nodes = 128
w1 = np.random.randn(x1.T.shape[0], no_of_hidden_nodes) 
w2 = np.random.randn(no_of_hidden_nodes, 10) 
b1 = np.zeros(no_of_hidden_nodes)
b2 = np.zeros(10)    
training_loss = list()
validation_loss = list()

    
for epoch in range(epochs):
  #Foward propogation
  z1 = np.matmul(x1, w1)+ b1
  a1 = sigmoid(z1)
  z2 = np.matmul(a1, w2)+ b2
  a2 = softmax(z2)
  final_loss = compute_loss(a2, y1)
  print("Loss for epoch", epoch+1,":", final_loss)
  training_loss.append(final_loss)
  
  #Back propogation
  del_z2 = a2 - y2
  del_w2 = (1/m) * np.dot(a1.T, del_z2)
  del_b2 = (1/m) * np.sum(del_z2,axis = 0, keepdims = True)
  
  del_a1 = np.dot(del_z2, w2.T)
  del_z1 = del_a1 * sigmoid_derivative(a1)
  del_w1 = (1/m) * np.dot(x1.T, del_z1)
  del_b1 = (1/m) * np.sum(del_z1, axis = 0, keepdims = True)
  
  w1 = w1 - learning_rate * del_w1
  b1 = b1 - learning_rate * del_b1
  w2 = w2 - learning_rate * del_w2
  b2 = b2 - learning_rate * del_b2

  z1_test = np.matmul(x2, w1)+ b1
  a1_test = sigmoid(z1_test)
  z2_test = np.matmul(a1_test, w2)+ b2
  a2_test = softmax(z2_test)
  final_loss = compute_loss(a2_test, y1_test)
  validation_loss.append(final_loss)

#Test data prediction
acc = 0
y_pred = list()
for i,j in itertools.zip_longest(normalized_x_test, y_test):
  z1 = np.dot(i, w1) + b1
  a1 = sigmoid(z1)            
  z2 = np.dot(a1, w2) + b2
  a2 = softmax(z2)
  prediction = np.argmax(a2)
  y_pred.append(prediction)
  if prediction == j:
    acc = acc+1
accuracy = (acc / normalized_x_test.shape[0])*100
print("Test Accuracy:", accuracy, "%")

#Generating confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#Plotting graphs
fig = plt.gcf()
plt.plot(training_loss, '--m', linestyle='--', markersize=1)
plt.plot(validation_loss, '--b', linestyle='--', markersize=1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
fig.savefig('training_validation_loss_task1.png', dpi=100)