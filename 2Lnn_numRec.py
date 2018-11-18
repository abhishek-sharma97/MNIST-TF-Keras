import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback
from matplotlib import pyplot as plt

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
trainData = mnist.train.images
trainLabels = np.array(mnist.train.labels, dtype=np.int32)
testData = mnist.test.images
testLabels = np.array(mnist.test.labels, dtype=np.int32)

print("training data shape: " + str(trainData.shape))
print("training labels shape: " + str(trainLabels.shape))
print("test data shape: " + str(testData.shape))
print("test labels shape: " + str(testLabels.shape))

image_size = 784
num_classes = 10

trainLabels = keras.utils.to_categorical(trainLabels, num_classes)
testLabels = keras.utils.to_categorical(testLabels, num_classes)

model = Sequential()

model.add(Dense(units=800, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))

model.summary()

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainData, trainLabels, batch_size=200, epochs=40, verbose=False, validation_split=.1, callbacks=[TestCallback((testData, testLabels))])
loss, accuracy  = model.evaluate(testData, testLabels, verbose=False)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

model2 = Sequential()

model2.add(Dense(units=800, activation='relu', input_shape=(image_size,)))
model2.add(Dense(units=num_classes, activation='softmax'))

model2.summary()

model2.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(trainData, trainLabels, batch_size=200, epochs=40, verbose=False, validation_split=.1, callbacks=[TestCallback((testData, testLabels))])
loss2, accuracy2  = model2.evaluate(testData, testLabels, verbose=False)

plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss2:.3}')
print(f'Test accuracy: {accuracy2:.3}')