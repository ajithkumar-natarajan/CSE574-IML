# -*- coding: utf-8 -*-
"""IML_Project3_task2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TaYIsq3WNzpTJiVc5zxdX2VIq3FxgIJl
"""

#Importing necessary libraries
import keras
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from sklearn.model_selection import train_test_split

#Loading the dataset directly from Keras' library; normalizing and reshaping it
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = (x_train.reshape(-1, 28, 28, 1)) / np.max(x_train)
x_test = (x_test.reshape(-1, 28, 28, 1)) / np.max(x_test)
np.max(x_train), np.max(x_test)
#Splitting the dataset into training and validation sets
train_set, val_set, train_out, val_out = train_test_split(x_train, x_train, test_size=0.2, random_state=42)

#Declaring model architecture
autoencoder = Sequential()

input_shape=(28, 28, 1)
filters=[32, 64, 128, 10]

if input_shape[0] % 8 == 0:
  pad3 = 'same'
else:
  pad3 = 'valid'

autoencoder.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=(28,28,1))) #28 x 28 x 32
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))
autoencoder.add(BatchNormalization())

autoencoder.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')) #14 x 14 x 64
autoencoder.add(BatchNormalization())
autoencoder.add(Flatten())

autoencoder.add(Dense(units=filters[3], name='embedding'))

autoencoder.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

autoencoder.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
autoencoder.add(BatchNormalization())

autoencoder.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))

#Declaring constants
batch_size = 32
epochs = 100
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))
num_classes = 10

#Compiling the model and settings loss function and optimizer
autoencoder.compile(loss='mean_squared_error', optimizer = Adam())

#Fitting the training data and validating the model
autoencoder_train = autoencoder.fit(train_set, train_out, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(val_set, val_out))

#Printing the summary of model
autoencoder.summary()

#Saving the model weights
autoencoder.save_weights('autoencoder_weights.h5')

#Loading the model weights
# autoencoder.load_weights('autoencoder_weights.h5')

#Plotting training & validation loss vs number of epochs while training for auto-encoder
import matplotlib.pyplot as plt
plt.plot(autoencoder.history.history['loss'], color='#009358', marker='o')
plt.plot(autoencoder.history.history['val_loss'], color='orange', marker='o')
plt.title("Training & validation loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Training loss', 'Validation loss'])
graph = plt.gcf()
plt.show()
graph.savefig('Task2_3_epoch_loss.png', dpi=300)

#Importing additional libraries required
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

#Separating the encoder layer and using the condensed representation
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('embedding').output)

#Declaring, fitting and predicting the clusters for the test dataset
kmeans = MiniBatchKMeans(n_clusters = 10, init= 'k-means++', n_init=20, random_state=42, verbose=1)
encoded = encoder.predict(x_test)
encoder_pred = encoder.predict(x_test).reshape(x_test.shape[0], -1)
y_pred = kmeans.fit_predict(encoder_pred)

#Observing the confusion matrix
#It can be observed that the clusters are not correctly aligned w.r.t the labels
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

confusionmatrix = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusionmatrix, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])
plt.figure(figsize = (10, 8))
plt.title('Original Confusion Matrix')
cm_original = plt.gcf()
fig = sns.heatmap(df_cm, annot=True, fmt='g')
cm_original.savefig('original_cm_kmeans.png', dpi=300)

#Using Hungarian algorithm to compute reordered confusion matrix
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

def _make_cost_m(cm):
    max_value = np.max(cm)
    return (- cm + max_value)

cm = confusion_matrix(y_test, y_pred)
indices = linear_assignment(_make_cost_m(cm))
number = [element[1] for element in sorted(indices, key=lambda x: x[0])]
reordered_cm = cm[:, number]

print("Accuracy of KMeans with auto-encoder: ", (np.trace(reordered_cm) / np.sum(reordered_cm))*100, "%")

#Visualizing reordered confusion matrix

import seaborn as sns

df_cm = pd.DataFrame(reordered_cm, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])
plt.figure(figsize = (10, 8))
plt.title('Reordered Confusion Matrix')
cm_reordered = plt.gcf()
fig = sns.heatmap(df_cm, annot=True, fmt='g')
# fig.pivot('Predicted labels', 'True labels')
cm_reordered.savefig('reordered_cm_kmeans.png', dpi=300)