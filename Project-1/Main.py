import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def normalize(df):
    result = df.copy()
    for column in df.columns:
        max_value = df[column].max()
        min_value = df[column].min()
        result[column] = (df[column] - min_value) / (max_value - min_value)
    return result

# reading csv file
pf = pd.read_csv("wdbc.csv")

#dropping first column and label
x=pf.iloc[:,2:]

#replacing labels
y = pf['label']
y = y.map({'B': 0, 'M': 1})


#normalizing dataset
x = normalize(x)


# partitioning dataset into train, validate and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)



X, Y = x_train.T, y_train.values.reshape(1, y_train.shape[0])
x_val, y_val = x_val.T, y_val.values.reshape(1, y_val.shape[0])


print(type(y_train))
epochs = 10000
learningrate = 0.01

def sigmoid(z):
 return 1 / (1 + np.exp(-z))


def logisticRegression(w, b, x, y, data):
    z = np.dot(w.T, X) + b
    p = sigmoid(z)
    cost = -np.sum(np.multiply(np.log(p), Y) + np.multiply((1 - Y), np.log(1 - p)))/m
    costlist.append(np.squeeze(cost))


costlist = []
m = X.shape[1]
w = np.random.randn(X.shape[0], 1)*0.01
b = 0
mselist = []
vallist = []
for epoch in range(epochs):
    z = np.dot(w.T, X) + b
    p = sigmoid(z)
    cost = -np.sum(np.multiply(np.log(p), Y) + np.multiply((1 - Y), np.log(1 - p)))/m
    costlist.append(np.squeeze(cost))
    dz = p-Y
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)
    w = w - learningrate * dw
    b = b - learningrate * db

    zl = np.dot(w.T, x_val) + b
    pl = sigmoid(zl)
    cost = -np.sum(np.multiply(np.log(pl), y_val) + np.multiply((1 - y_val), np.log(1 - pl)))/m
    vallist.append(np.squeeze(cost))

    error = []
    error = np.subtract(Y, p)
    we_squared = np.square(error)
    mse = np.square(np.subtract(Y,p)).mean()
    mselist.append(mse)

plt.figure(1)
plt.ylabel('cost')
plt.xlabel('Epochs')

plt.plot(costlist, color='red')
plt.figure(3)
plt.plot(vallist, color='blue')

plt.figure(2)
plt.ylabel('Mean Squared Error')
plt.xlabel('Epochs')

plt.plot(mselist)
# print(costlist)


#validation
z = np.dot(x_val.T, w) + b
p = sigmoid(z)

#model evaluation metrics - mean squared error
error = []
y_val = y_val[np.newaxis,:]
y_val = y_val.T
error = np.subtract(y_val, p)
we_squared = np.square(error)
mse = np.square(np.subtract(y_val,p)).mean()
print('Mean Squared Error', mse)
print('------')

#test data
z = np.dot(x_test, w) + b
p = sigmoid(z)

output = []
for i in range(len(p)):
    if(p[i] <= 0.5):
        output.append(0)
    else:
        output.append(1)

#prediction, accuracy, recall
actual = y_test.tolist()
predicted = output


print('Accuracy Score :', accuracy_score(actual, predicted))
confusion_matrix = confusion_matrix(actual, predicted)
print('Confusion Matrix:', confusion_matrix)

tp = confusion_matrix[0][0]
fn = confusion_matrix[0][1]
fp = confusion_matrix[1][0]
tn = confusion_matrix[1][1]
precision = tp/(tp+fp)
print('Precision', precision)

recall = tp/(tp+fn)
print('Recall', recall)


accuracy = (tp + tn) / (tp + tn + fp + fn)
print('Accuracy', accuracy)

print('Report : ')
print(classification_report(actual, predicted))
