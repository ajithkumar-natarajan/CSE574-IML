# -*- coding: utf-8 -*-
# Read Fashion MNIST dataset

import util_mnist_reader as mnist_reader
X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

print(X_train)
print(y_train)

# Your code goes here . . .
