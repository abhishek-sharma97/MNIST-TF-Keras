import numpy as np
import tensorflow as tf
from sklearn import svm, metrics
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
trainData = mnist.train.images
trainLabels = np.array(mnist.train.labels, dtype=np.int32)
testData = mnist.test.images
testLabels = np.array(mnist.test.labels, dtype=np.int32)

print("training data shape: " + str(trainData.shape))
print("training labels shape: " + str(trainLabels.shape))
print("test data shape: " + str(testData.shape))
print("test labels shape: " + str(testLabels.shape))

C = pow(10,6)
gamma = pow(10,-4)/784
print(gamma)

numPred = svm.SVC(kernel='rbf', random_state=0, gamma=gamma, C=C)

numPred.fit(trainData, trainLabels)

predRes = numPred.predict(testData)

print("Classification report for classifier %s:n%sn" % (numPred, metrics.classification_report(testLabels, predRes)))