
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np

# Import helper functions
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.utils import train_test_split, to_categorical, normalize
from mlfromscratch.utils import get_random_subsets, shuffle_data, Plot
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D
from mlfromscratch.deep_learning.layers import AveragePooling2D, ZeroPadding2D, BatchNormalization, RNN


def main_rnn():

    optimizer = Adam()

    def gen_mult_ser(nums):
        """ Method which generates multiplication series """
        X = np.zeros([nums, 10, 61], dtype=float)
        y = np.zeros([nums, 10, 61], dtype=float)
        for i in range(nums):
            start = np.random.randint(2, 7)
            mult_ser = np.linspace(start, start*10, num=10, dtype=int)
            X[i] = to_categorical(mult_ser, n_col=61)
            y[i] = np.roll(X[i], -1, axis=0)
        y[:, -1, 1] = 1 # Mark endpoint as 1
        return X, y


    def gen_num_seq(nums):
        """ Method which generates sequence of numbers """
        X = np.zeros([nums, 10, 20], dtype=float)
        y = np.zeros([nums, 10, 20], dtype=float)
        for i in range(nums):
            start = np.random.randint(0, 10)
            num_seq = np.arange(start, start+10)
            X[i] = to_categorical(num_seq, n_col=20)
            y[i] = np.roll(X[i], -1, axis=0)
        y[:, -1, 1] = 1 # Mark endpoint as 1
        return X, y

    X, y = gen_mult_ser(3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # Model definition
    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy)
    clf.add(RNN(10, activation="tanh", bptt_trunc=5, input_shape=(10, 61)))
    clf.add(Activation('softmax'))
    clf.summary("RNN")

    # Print a problem instance and the correct solution
    tmp_X = np.argmax(X_train[0], axis=1)
    tmp_y = np.argmax(y_train[0], axis=1)
    print ("Number Series Problem:")
    print ("X = [" + " ".join(tmp_X.astype("str")) + "]")
    print ("y = [" + " ".join(tmp_y.astype("str")) + "]")
    print ()

    train_err, _ = clf.fit(X_train, y_train, n_epochs=500, batch_size=512)

    # Predict labels of the test data
    y_pred = np.argmax(clf.predict(X_test), axis=2)
    y_test = np.argmax(y_test, axis=2)

    print ()
    print ("Results:")
    for i in range(5):
        # Print a problem instance and the correct solution
        tmp_X = np.argmax(X_test[i], axis=1)
        tmp_y1 = y_test[i]
        tmp_y2 = y_pred[i]
        print ("X      = [" + " ".join(tmp_X.astype("str")) + "]")
        print ("y_true = [" + " ".join(tmp_y1.astype("str")) + "]")
        print ("y_pred = [" + " ".join(tmp_y2.astype("str")) + "]")
        print ()
    
    accuracy = np.mean(accuracy_score(y_test, y_pred))
    print ("Accuracy:", accuracy)

    training = plt.plot(range(500), train_err, label="Training Error")
    plt.title("Error Plot")
    plt.ylabel('Training Error')
    plt.xlabel('Iterations')
    plt.show()

def main_cnn():

    #----------
    # Conv Net
    #----------

    optimizer = Adam()

    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape((-1,1,8,8))
    X_test = X_test.reshape((-1,1,8,8))

    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    clf.add(Conv2D(n_filters=16, filter_shape=(3,3), stride=1, input_shape=(1,8,8), padding='same'))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Conv2D(n_filters=32, filter_shape=(3,3), stride=1, padding='same'))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Flatten())
    clf.add(Dense(256))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.4))
    clf.add(BatchNormalization())
    clf.add(Dense(10))
    clf.add(Activation('softmax'))

    print ()
    clf.summary(name="ConvNet")

    train_err, val_err = clf.fit(X_train, y_train, n_epochs=50, batch_size=256)

    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, accuracy = clf.test_on_batch(X_test, y_test)
    print ("Accuracy:", accuracy)


    y_pred = np.argmax(clf.predict(X_test), axis=1)
    X_test = X_test.reshape(-1, 8*8)
    # Reduce dimension to 2D using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Convolutional Neural Network", accuracy=accuracy, legend_labels=range(10))

if __name__ == "__main__":
    main_cnn()





