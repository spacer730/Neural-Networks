'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

#The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Sequential is a normal MLP type structure. Dense implents operation: output = activation(dot(input, kernel) + bias)
#Input shape is 784, because 28x28 pixels for the hand written digits
#Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#Summary prints a summary of your model
model.summary()

#Configure the learning process before training the model. Here we specify which loss function, optimizer and a list of metrics to use.
model.compile(loss='mse',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#Here we train the model. An epoch is one forward pass and one backward pass of all the training examples.
#The batch size is the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
#Example: 1000 training examples, batchsize is 500 then it takes 2 iterations for one epoch to complete.
#Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. (How to see the training process for each epoch)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

#Returns the loss value & metrics values for the model in test mode.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Misaccurate digits:', score[2])
