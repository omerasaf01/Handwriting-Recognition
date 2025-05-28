import tensorflow as ts
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import os

# This command automatically downloads and splits the data into a training set and a test set.
# The training set contains 60,000 images, while the test set contains 10,000.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32")  # 255
x_test = x_test.astype("float32")

x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

if __name__ == "__main__":
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Accuracy:", test_accuracy)
