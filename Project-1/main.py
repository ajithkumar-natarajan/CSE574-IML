# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KAOsDq-OanMV1FZuemgq76OWpWRzCPXP
"""

#Importing necessary libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix


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


"""Does the process of logistic regression. Takes the generic equation, estimates the probability of the cancer being malignant for the instance passed. The loss is evaluated which is subsequently used for updating the weights and bias."""
def logistic_regression(x, y, w, b, learning_rate):
    m = x.shape[1]
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    loss = -np.sum(np.multiply(np.log(a), y) + np.multiply((1 - y), np.log(1 - a)))/m
    dz = a-y #Error in prediction is calculated.
    dw = (1 / m) * np.dot(x, dz.T)
    db = (1 / m) * np.sum(dz)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return loss, w, b, a

# Reading the dataset  
dataset = pd.read_csv("wdbc.dataset", names=["ID", "Label", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24", "X25", "X26", "X27", "X28", "X29", "X30"])

#Dropping patient ID and labels
x = dataset.iloc[:, 2:] 

#Assigning label to y and mapping benign as '0' and malignant cases as '1'
y = dataset['Label']
y = y.map({'B': 0, 'M': 1})

#Normalizing the dataset
x = normalize(x)

#Splitting the dataset into training (80%), validation (10%) and test (10%) data.
x_train, x_remaining, y_train, y_remaining = train_test_split(x, y, train_size=0.8, random_state=10)
x_val, x_test, y_val, y_test = train_test_split(x_remaining, y_remaining, test_size=0.5, random_state=10)


#Flipping the data
x_train, y_train = x_train.T, y_train.values.reshape(1, y_train.shape[0])
x_val, y_val = x_val.T, y_val.values.reshape(1, y_val.shape[0])

#Initializing parameters
learning_rate = 10
w = np.random.randn(x_train.shape[0], 1)*0.1
b = 0
training_loss = []
validation_loss = []
# training_accuracy = []
# val_accuracy = []
# output = []
# training_prediction = []
# val_prediction = []
# training_accuracy = []
# val_accuracy = []

#Training model over 10000 epochs and calculating training loss and validation loss to study the training process
for epoch in range(10000):
    loss, w, b, a  = logistic_regression(x_train, y_train, w, b, learning_rate)
    training_loss.append(np.squeeze(loss))
#     aList = a.tolist()
#     print((a[0][10]))
#     y_train_list = y_train.tolist()
#     for i in range(len(aList)):
#       if(aList[0][i] <= 0.5):
#           training_prediction.append(0)
#       else:
#           training_prediction.append(1)
# #       print(aList[0][i])
          
#     count = 0
#     for i in range(len(y_train_list)):
#       if(y_train_list[i] == training_prediction[i]):
#         count +=1
#     acc = 100*count/y_train.shape[0]
#     training_accuracy.append(acc)
    
    z = np.dot(w.T, x_val) + b
    a = sigmoid(z)
    m = x_val.shape[1]
    loss = -np.sum(np.multiply(np.log(a), y_val) + np.multiply((1 - y_val), np.log(1 - a)))/m
    validation_loss.append(np.squeeze(loss))
#     aList = a.tolist()
#     y_val_list = y_val.tolist()
#     for i in range(len(aList)):
#       if(aList[0][i] <= 0.5):
#           val_prediction.append(0)
#       else:
#           val_prediction.append(1)
#     count = 0
#     for i in range(len(y_val_list)):
#       if(y_val_list[i] == val_prediction[i]):
#         count +=1
#     acc = 100*count/y_val.shape[0]
#     val_accuracy.append(acc)      


#Loss vs epochs for training and validation data is plotted to observe the convergence
plot.ylabel('Loss')
plot.xlabel('Epochs')
plot.title('Loss vs Epochs')
plot.plot(training_loss, label='Training dataset', color='red')
plot.plot(validation_loss, label='Validation dataset', color='blue')
plot.legend(loc='best')
# print(validation_loss[0])
# print(validation_loss[-1])
# print(training_loss[0])
# print(training_loss[-1])
# plt.savefig("O1.svg")
# print((training_accuracy))
# plot.ylabel('Accuracy')
# plot.xlabel('Epochs')
# plot.title('Accuracy vs Epochs')
# plot.plot(training_accuracy, label='Training dataset', color='red')
# plot.plot(val_accuracy, label='Validation dataset', color='blue')
# plot.legend(loc='best')
# plt.savefig("O1.svg")


#Testing the model with unseen data to evaluate it
z = np.dot(x_test, w) + b
a = sigmoid(z)

#Convert the probabilistic output to class
for i in range(len(a)):
    if(a[i] <= 0.5):
        output.append(0)
    else:
        output.append(1)

#Computation of evaluation parameters
y_test = y_test.tolist()
predicted_labels = output

confusion_matrix = confusion_matrix(actual, predicted)
print(confusion_matrix)

tp = confusion_matrix[0][0]
fp = confusion_matrix[0][1]
fn = confusion_matrix[1][0]
tn = confusion_matrix[1][1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
print('Accuracy:', accuracy)

precision = tp/(tp+fp)
print('Precision:', precision)

recall = tp/(tp+fn)
print('Recall:', recall)