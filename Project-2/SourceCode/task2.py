# -*- coding: utf-8 -*-
"""CSE574-Project2-task2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GnlS5tz5u5hOYDXMDLZpG_TZZw7h-6PT
"""

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
# import pickle
from sklearn.metrics import classification_report, confusion_matrix


# def load_mnist(path, kind='train'):
# 	import os
# 	import gzip
# 	import numpy as np

# 	"""Load MNIST data from `path`"""
# 	labels_path = os.path.join(path,
#                                '%s-labels-idx1-ubyte.gz'
#                                % kind)
# 	images_path = os.path.join(path,
#                                '%s-images-idx3-ubyte.gz'
#                                % kind)
# 	with gzip.open(labels_path, 'rb') as lbpath:
# 		labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
#                                offset=8)

# 	with gzip.open(images_path, 'rb') as imgpath:
# 		images = np.frombuffer(imgpath.read(), dtype=np.uint8,
#                                offset=16).reshape(len(labels), 784)

# 	return images, labels

# x_train, y_train = load_mnist('data/fashion', kind='train')
# x_test, y_test = load_mnist('data/fashion', kind='t10k')

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

batch_size = 512

from keras.callbacks import TensorBoard



# print(x_train.shape)
# print(x_test.shape)
# print(type(x_train))

# x_train = x_train.reshape(x_train, (60000, 28, 28 ))
# x_test = x_test.reshape(x_test, (60000, 28, 28))

# print(x_train.shape)
# print(x_test.shape)

# i = 13
# plt.imshow(x_train[i,:,:], cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()
# print("label-> ", y_train[i])

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test_orig = y_test
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), 
                          # keras.layers.Dense(400, activation='sigmoid'), 
                          keras.layers.Dense(128, activation='relu'), 
                          keras.layers.Dense(10, activation='softmax')])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=batch_size,
                         write_images=True)

model.fit(x_train, y_train, epochs=50, batch_size=batch_size, validation_data=(x_test, y_test))
# with open('trainHistoryDict', 'wb') as file_pi:
        # pickle.dump(output.history, file_pi)

output = model.predict_classes(x_test)
# con_mat = tf.math.confusion_matrix(labels=y_test, predictions=output).numpy()
# con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
 
# con_mat_df = pd.DataFrame(con_mat_norm,
#                      index = classes, 
#                      columns = classes)
confusion_matrix = confusion_matrix(y_test_orig, output)

print(confusion_matrix)

!unzip data_with_notebook.zip

!rm -r scripts/