"""Convolutional Neural Network layer structure"""
from keras import models
from keras import layers

def get_model(n_classes):
    # building a linear stack of layers with the sequential model
    model = models.Sequential()

    # convolutional layer
    model.add(layers.Conv2D(8, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)))


    # flatten output of conv
    model.add(layers.Flatten())

    # hidden layer
    model.add(layers.Dense(512, activation='relu'))

    # output layer
    model.add(layers.Dense(n_classes, activation='softmax'))

    # compiling the sequential model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


