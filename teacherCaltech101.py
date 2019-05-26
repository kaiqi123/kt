import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np

class MentorForCaltech101(object):

	def __init__(self, trainable=True):
		self.trainable = trainable
		self.data_dict = np.load("vgg16.npy").item()
		#self.dropout = dropout
		#self.parameters = []

	def build_convLayer(self, input, layerName):
		with tf.name_scope(layerName) as scope:
			kernel = tf.Variable(self.data_dict[layerName][0], name='weights', trainable=self.trainable)
			biases = tf.Variable(self.data_dict[layerName][1], name='biases', trainable=self.trainable)
			conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
			out = tf.nn.bias_add(conv, biases)
			mean, var = tf.nn.moments(out, axes=[0])
			batch_norm = (out - mean) / tf.sqrt(var + tf.Variable(1e-10))
			relu = tf.nn.relu(batch_norm, name=scope)
			return relu

	def build_fcLayer(self, input, layerName, weightsDictName=None, num_classes=None):
		with tf.name_scope(layerName):
			if layerName == "fc1" or layerName == "fc2":
				fc_weights = tf.Variable(self.data_dict[weightsDictName][0], name="weights", trainable=self.trainable)
				fc_biases = tf.Variable(self.data_dict[weightsDictName][1], name="biases", trainable=self.trainable)
				shape = int(np.prod(input.get_shape()[1:]))
				input_flat = tf.reshape(input, [-1, shape])
				out = tf.nn.bias_add(tf.matmul(input_flat, fc_weights), fc_biases)
				out = tf.nn.relu(out)
				print(shape)
			elif layerName == "fc3":
				fc_weights = tf.Variable(tf.truncated_normal([4096, num_classes], dtype=tf.float32, stddev=1e-2),
								   name='weights', trainable=self.trainable)
				fc_biases = tf.Variable(tf.constant(1.0, shape=[num_classes], dtype=tf.float32),
								   name='biases', trainable=self.trainable)
				out = tf.nn.bias_add(tf.matmul(input, fc_weights), fc_biases)
			return out

	def build_vgg16_teacher(self, rgb, num_classes, temp_softmax):
		with tf.name_scope('mentor'):
			self.conv1_1 = self.build_convLayer(rgb, "conv1_1")
			self.conv1_2 = self.build_convLayer(self.conv1_1, "conv1_2")
			self.pool1 = tf.nn.max_pool(self.conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool1')

			self.conv2_1 = self.build_convLayer(self.pool1, "conv2_1")
			self.conv2_2 = self.build_convLayer(self.conv2_1, "conv2_2")
			self.pool2 = tf.nn.max_pool(self.conv2_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')

			self.conv3_1 = self.build_convLayer(self.pool2, "conv3_1")
			self.conv3_2 = self.build_convLayer(self.conv3_1, "conv3_2")
			self.conv3_3 = self.build_convLayer(self.conv3_2, "conv3_3")
			self.pool3 = tf.nn.max_pool(self.conv3_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool3')

			self.conv4_1 = self.build_convLayer(self.pool3, "conv4_1")
			self.conv4_2 = self.build_convLayer(self.conv4_1, "conv4_2")
			self.conv4_3 = self.build_convLayer(self.conv4_2, "conv4_3")
			self.pool4 = tf.nn.max_pool(self.conv4_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool4')

			self.conv5_1 = self.build_convLayer(self.pool4, "conv5_1")
			self.conv5_2 = self.build_convLayer(self.conv5_1, "conv5_2")
			self.conv5_3 = self.build_convLayer(self.conv5_2, "conv5_3")
			self.pool5 = tf.nn.max_pool(self.conv5_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool5')

			self.fc1 = self.build_fcLayer(self.pool5, "fc1", weightsDictName="fc6")
			self.fc2 = self.build_fcLayer(self.fc1, "fc2", weightsDictName="fc7")
			self.fc3 = self.build_fcLayer(self.fc2, "fc3", weightsDictName=None, num_classes=num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			return self

	# Variables of the last layer of the teacher.
	def get_training_vars(self):
		training_vars = []
		independent_weights = [var for var in tf.global_variables() if var.op.name == "mentor/fc3/weights"]
		independent_biases = [var for var in tf.global_variables() if var.op.name == "mentor/fc3/biases"]
		training_vars.append(independent_weights)
		training_vars.append(independent_biases)
		print("independent_weights(mentor/fc3/weights): ", independent_weights)
		print("independent_biases(mentor/fc3/biases): ", independent_biases)
		return training_vars

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc3, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')

	def training(self, loss, learning_rate_pretrained, learning_rate_for_last_layer, global_step, variables_to_restore, train_last_layer_variables):
		### Adding Momentum of 0.9
		optimizer1 = tf.train.AdamOptimizer(learning_rate_pretrained)
		optimizer2 = tf.train.AdamOptimizer(learning_rate_for_last_layer)
		train_op1 = optimizer1.minimize(loss, global_step=global_step, var_list = variables_to_restore)
		train_op2 = optimizer2.minimize(loss, global_step=global_step, var_list = train_last_layer_variables)
		return tf.group(train_op1, train_op2)
