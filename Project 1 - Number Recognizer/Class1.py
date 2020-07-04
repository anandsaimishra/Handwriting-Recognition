#  Property of Godslayer™
#  Code wirtten by Anand Sai Mishra
#  On : 7/3/20, 4:58 PM
import matplotlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
'''tf.logging.set_verbosity(tf.logging.ERROR)'''
print('Using Tensorflow version', tf.__version__)
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train shape:', x_train)
print('y_train shape:', y_train)
print('x_test shape:', x_test)
print('y_test shape:', y_test)
from matplotlib import pyplot as plt

plt.imshow(x_train[0], cmap='binary')
plt.show()
print("This digit is :",y_train[0])
print("This classifier can differentiale between these digits: ", set(y_train))
from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
print("y_test_encoded: ",y_test_encoded.shape)
print("y_train_encoded: ",y_train_encoded.shape)
print("the same digit(5) encoded is: ", y_train_encoded[0])

"""
Preprocessing the Examples
Right now the shape of our inputs is 28x28 and we need to convert it into a 10 input
"""

x_train_reshaped = np.reshape(x_train, (60000, 784)) #each value in this vector is now a 784 dimentional vector
x_test_reshaped = np.reshape(x_test, (10000, 784))


print("x_test_reshaped :", set(x_test_reshaped[0]))
print("x_train_reshaped : ", set(x_train_reshaped[0]))

'''
Data Normalization : In order to normalize the data we need to calculate the mean and median of the dataset.

'''

x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
epsilon = 1e-10

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

print("Normalized Test Set :",x_test_norm)
print("Normalized Training Set:", x_train_norm)

'''
Creating a neural network model - using keras
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),#Input of this layer is output of the previous layer
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics='accuracy'
)

print(model.summary())

'''
Training the model. We shall train the model for three epoc's 
'''

model.fit(x_train_norm, y_train_encoded, epochs=3)
_, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test Set Accuracy: ', accuracy * 100)

'''
Predictions
'''
preds = model.predict(x_test_norm)
print('Shape of Preds :', preds.shape)

'''
Plot the Results
'''
plt.figure(figsize=(12,12))

start_index = 0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred = np.argmax(preds[start_index+i])
    gt = y_test[start_index+i]

    col = 'g'
    if pred != gt:
        col = 'r'

    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, pred, gt), color=col)
    plt.imshow(x_test[start_index+i], cmap='binary')
    plt.show()