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

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
#Change loss to 'mse' to use MSE error
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = np.array([0,1,2,3,4,5,6,7,8,9])
y_pred = model.predict_classes(x_test)
y_test_cnf = [np.argmax(elt) for elt in y_test]

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_cnf, y_pred)

total = np.sum(cnf_matrix, axis=1)
difference = [total[i] - cnf_matrix[i,i] for i in range(10)]
percentage_misclassified = 100*(difference/total)

numberone = np.argmax(percentage_misclassified)

percentage_misclassified2 = 100*(difference/total)
percentage_misclassified2[numberone] = 0 
numbertwo = np.argmax(percentage_misclassified2)

percentage_misclassified3 = 100*(difference/total)
percentage_misclassified3[numberone] = 0
percentage_misclassified3[numbertwo] = 0
numberthree = np.argmax(percentage_misclassified3)

print('')
print('Top 3 misclassified digits are:')
print('The #1 most misclassified digit is: ' + str(numberone) + ' with a percentage of: ' +str(percentage_misclassified[numberone]) )
print('The #2 most misclassified digit is: ' + str(numbertwo) + ' with a percentage of: ' +str(percentage_misclassified[numbertwo]) )
print('The #3 most misclassified digit is: ' + str(numberthree) + ' with a percentage of: ' +str(percentage_misclassified[numberthree]) )

# Plot non-normalized confusion matrix
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()
