from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnnModel(features, labels, mode):
	""" 
	CNN model function -
	* mode - TRAIN, EVAL, PREDICT - from tf.estimator.ModeKeys
	"""
	# Input Layer
	# Reshape X into 4D tensor - MNIST images => 28x28 grayscale
	ipLayer = tf.reshape(features["x"], [-1,28,28,1])

	# Convolution Layer 1 - 32 5x5 filters with ReLU activation and "SAME" padding technique
	# Earlier implementations of LeNet used sigmoid and tanh activations
	conv1 = tf.layers.conv2d(inputs=ipLayer, 
		filters=32, 
		kernel_size=[5, 5], 
		padding="same", 
		activation=tf.nn.relu)

	# MaxPooling Layer 1 - 2x2 filters with strides of 2
	maxPool1 = tf.layers.max_pooling2d(inputs=conv1, 
		pool_size=[2, 2], 
		strides=2)

	# Convolution Layer 2 - 64 5x5 filters with ReLU activation and "SAME" padding technique
	conv2 = tf.layers.conv2d(inputs=maxPool1, 
		filters=64, 
		kernel_size=[5, 5], 
		padding="same", 
		activation=tf.nn.relu)

	# MaxPooling Layer 2 - 2x2 filters with strides of 2
	maxPool2 = tf.layers.max_pooling2d(inputs=conv2, 
		pool_size=[2, 2], 
		strides=2)

	# Fully Connected Layer 1 - 1024 neurons with ReLU activation, with dropout regularisation rate of 0.4
	# Flattening the last layer into an array
	maxPool2Flat = tf.reshape(maxPool2, [-1,7*7*64])
	fc1 = tf.layers.dense(inputs=maxPool2Flat, 
		units=1024, 
		activation=tf.nn.relu)
	dropoutReg = tf.layers.dropout(inputs=fc1, 
		rate=0.4, 
		training=mode == tf.estimator.ModeKeys.TRAIN)

	# Fully Connected Layer 2 - 10 neurons for each digits in 0-9
	fc2 = tf.layers.dense(inputs=dropoutReg, units=10)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=fc2, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
		"probabilities": tf.nn.softmax(fc2, name="softmax_tensor")
		}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Cost calculation for both TRAIN and EVAL methods
	cost = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=fc2)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		trainOps = optimizer.minimize(loss=cost, 
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=cost, train_op=trainOps)

	evalMetrics = {
		"accuracy": tf.metrics.accuracy(labels=labels, 
			predictions=predictions["classes"])
	}
	return tf.estimator.EstimatorSpec(mode=mode, loss=cost, eval_metric_ops=evalMetrics)

def main(unused_argv):
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	trainData = mnist.train.images
	trainLabels = np.array(mnist.train.labels, dtype=np.int32)
	testData = mnist.test.images
	testLabels = np.array(mnist.test.labels, dtype=np.int32)
	indices = np.random.randint(0,testData.shape[0],5)
	predData = testData[indices]
	predLabels = testLabels[indices]
	# Estimator Creation
	numPred = tf.estimator.Estimator(model_fn=cnnModel, 
		model_dir="/tmp/mdl_cps")

	# Setup logging for predictions
	tensorLog = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensorLog, every_n_iter=50)

	# Training model
	trainIpFunc = tf.estimator.inputs.numpy_input_fn(x={"x": trainData}, 
		y=trainLabels, 
		batch_size=100, 
		num_epochs=None, 
		shuffle=True)
	numPred.train(input_fn=trainIpFunc, 
		steps=7000, 
		hooks=[logging_hook])

	# Test and evaluate model
	testIpFunc = tf.estimator.inputs.numpy_input_fn(x={"x": testData}, 
		y=testLabels,  
		num_epochs=1, 
		shuffle=False)

	testRes = numPred.evaluate(input_fn=testIpFunc)
	predIpFunc = tf.estimator.inputs.numpy_input_fn(x={"x": predData}, 
		num_epochs=1, 
		shuffle=False)
	predRes = list(numPred.predict(predIpFunc))
	print(predRes)
	print(predLabels)

if __name__ == '__main__':
	tf.app.run()