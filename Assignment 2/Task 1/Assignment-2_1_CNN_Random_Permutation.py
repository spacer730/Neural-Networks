'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])

def permute_array(array, permutation_order, row_order):
	permutation = np.zeros(np.shape(array))

	for i in range(np.shape(array)[0]):
		for j in range(np.shape(array)[1]):
			permutation[i][j] = array[row_order[i]][permutation_order[i][j]]

	return permutation

permutation_order = np.arange(28)
permutation_order = [np.random.permutation(permutation_order) for i in range(28)]
which_row = np.random.permutation(np.arange(28))

#Permute the input arrays
x_train = np.array([permute_array(x_train[i], permutation_order, which_row) for i in range(60000)])
x_test = np.array([permute_array(x_test[i], permutation_order, which_row) for i in range(10000)])

print(x_train[0])

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#Change loss to keras.losses.mse to use MSE error function
model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
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
