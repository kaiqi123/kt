import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K
"""
Removed all kinds of regularizers such as dropout and batch_normalization
"""

class Mentee(object):

	def __init__(self, seed):
		self.parameters = []
		self.seed = seed

	def fc_student(self, input, layerName, out_filter):
		with tf.name_scope(layerName):
			shape = int(np.prod(input.get_shape()[1:]))
			fc_weights = tf.Variable(tf.truncated_normal([shape, out_filter], dtype=tf.float32, stddev=1e-2),trainable=True, name='weights')
			fc_biases = tf.Variable(tf.constant(1.0, shape=[out_filter], dtype=tf.float32), trainable=True,name='biases')
			input_flat = tf.reshape(input, [-1, shape])
			fc = tf.nn.bias_add(tf.matmul(input_flat, fc_weights), fc_biases)
			relu = tf.nn.relu(fc)
			if layerName == "fc3":
				return fc
			else:
				return relu

	def build_student_oneConvLayer(self, input, layerName, out_filter):
		with tf.name_scope(layerName):
			num_filters_in = int(input.shape[3])
			kernel = tf.Variable(tf.truncated_normal([3, 3, num_filters_in, out_filter], dtype=tf.float32, stddev=1e-2, seed=self.seed),trainable=True, name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[out_filter], dtype=tf.float32), trainable=True, name='biases')
			conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
			out = tf.nn.bias_add(conv, biases)
			out = tf.nn.relu(out, name="relu")
			return out

	def build_student_conv6fc3(self, images, num_classes, temp_softmax):
		K.set_learning_phase(True)
		with tf.name_scope('mentee'):
			conv1_1 = self.build_student_oneConvLayer(images, "conv1_1", 64)
			pool1 = tf.nn.max_pool(conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			conv2_1 = self.build_student_oneConvLayer(pool1, "conv2_1", 128)
			pool2 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			conv3_1 = self.build_student_oneConvLayer(pool2, "conv3_1", 256)
			pool3 = tf.nn.max_pool(conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			conv4_1 = self.build_student_oneConvLayer(pool3, "conv4_1", 512)
			pool4 = tf.nn.max_pool(conv4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			conv5_1 = self.build_student_oneConvLayer(pool4, "conv5_1", 512)
			pool5 = tf.nn.max_pool(conv5_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
			print(pool5)

			conv6_1 = self.build_student_oneConvLayer(pool5, "conv6_1", 512)
			pool6 = tf.nn.max_pool(conv6_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool6')
			print(pool6)

			fc1 = self.fc_student(pool6, "fc1", 4096)
			fc2 = self.fc_student(fc1, "fc2", 4096)
			self.fc3 = self.fc_student(fc2, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(fc1)
			print(fc2)
			print(self.fc3)
			return self

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.fc3, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')

	def training(self, loss, learning_rate, global_step):
		# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True)
		# optimizer = tf.train.AdamOptimizer(learning_rate)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def _calc_num_trainable_params(self):
		self.num_trainable_params = np.sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
		print('number of trainable params: ' + str(self.num_trainable_params))

	def build_conv5fc2(self, rgb, num_classes, temp_softmax, seed,train_mode):

		print("build student conv5fc2")
		K.set_learning_phase(True)

		# conv1_1
		with tf.name_scope('mentee_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, self.num_channels, 64], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			#out = self.extra_regularization(out)

			self.conv1_1 = tf.nn.relu(out, name=scope)
			#self.conv1_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv1_1')(self.conv1_1)
			self.parameters += [kernel, biases]

		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')
		#conv2_1
		with tf.name_scope('mentee_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
			#self.conv2_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv2_1')(self.conv2_1)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')

		with tf.name_scope('mentee_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
			#self.conv3_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv3_1')(self.conv3_1)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv3_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool3')

		with tf.name_scope('mentee_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
													 stddev=1e-2, seed= seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_1 = tf.nn.relu(out, name=scope)
			#self.conv4_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv4_1')(self.conv4_1)
			self.parameters += [kernel, biases]

		self.pool4 = tf.nn.max_pool(self.conv4_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool4')

		with tf.name_scope('mentee_conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv5_1 = tf.nn.relu(out, name=scope)
			#self.conv5_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv5_1')(self.conv5_1)
			self.parameters += [kernel, biases]

		self.pool5 = tf.nn.max_pool(self.conv5_1,
							ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1],
							padding='SAME',
							name='pool5')

		# fc1
		with tf.name_scope('mentee_fc1') as scope:
			shape = int(np.prod(self.pool5.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
														 dtype=tf.float32, stddev=1e-2, seed = seed), trainable = self.trainable,name='mentee_weights')
			fc1b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			pool5_flat = tf.reshape(self.pool5, [-1, shape])
			fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(fc1l)
			#self.fc1 = BatchNormalization(axis = -1, name= 'mentee_bn_fc1')(self.fc1)
			#if train_mode == True:
			#print("Traine_mode is true")
			#self.fc1 = tf.nn.dropout(self.fc1, 0.5, seed = seed)
			self.parameters += [fc1w, fc1b]

		with tf.name_scope('mentee_fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([4096, num_classes],
														 dtype=tf.float32, stddev=1e-2, seed = seed), trainable = self.trainable,name='mentee_weights')
			fc3b = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			self.fc3l = tf.nn.bias_add(tf.matmul(self.fc1, fc3w), fc3b)
			#self.fc3 = tf.nn.relu(fc3l)
			self.parameters += [fc3w, fc3b]

		self.softmax = tf.nn.softmax(self.fc3l/temp_softmax)
		return self

	def build_conv4fc1(self, rgb, num_classes, temp_softmax, seed,train_mode):

		print("build student conv4fc1")

		K.set_learning_phase(True)

		# conv1_1
		with tf.name_scope('mentee_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, self.num_channels, 64], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			#out = self.extra_regularization(out)

			self.conv1_1 = tf.nn.relu(out, name=scope)
			#self.conv1_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv1_1')(self.conv1_1)
			self.parameters += [kernel, biases]

		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')
		#conv2_1
		with tf.name_scope('mentee_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
			#self.conv2_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv2_1')(self.conv2_1)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')

		with tf.name_scope('mentee_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
			#self.conv3_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv3_1')(self.conv3_1)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv3_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool3')

		with tf.name_scope('mentee_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
													 stddev=1e-2, seed= seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_1 = tf.nn.relu(out, name=scope)
			#self.conv4_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv4_1')(self.conv4_1)
			self.parameters += [kernel, biases]

		self.pool4 = tf.nn.max_pool(self.conv4_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool4')

		with tf.name_scope('mentee_fc3') as scope:
			shape = int(np.prod(self.pool4.get_shape()[1:]))
			fc3w = tf.Variable(tf.truncated_normal([shape, num_classes],
												   dtype=tf.float32, stddev=1e-2, seed=seed), trainable=self.trainable,
							   name='mentee_weights')
			fc3b = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
							   trainable=self.trainable, name='mentee_biases')
			pool1_flat = tf.reshape(self.pool4, [-1, shape])
			self.fc3l = tf.nn.bias_add(tf.matmul(pool1_flat, fc3w), fc3b)
			# self.fc3 = tf.nn.relu(fc3l)
			self.parameters += [fc3w, fc3b]

		self.softmax = tf.nn.softmax(self.fc3l / temp_softmax)
		return self

	def build_conv3fc1(self, rgb, num_classes, temp_softmax, seed, train_mode):

		print("independent student build_conv3fc1")

		K.set_learning_phase(True)
		# conv1_1
		with tf.name_scope('mentee_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, self.num_channels, 64], dtype=tf.float32,
													 stddev=1e-2, seed=seed), trainable=self.trainable,
								 name='mentee_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			# out = self.extra_regularization(out)

			self.conv1_1 = tf.nn.relu(out, name=scope)
			# self.conv1_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv1_1')(self.conv1_1)
			self.parameters += [kernel, biases]

		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')
		# conv2_1
		with tf.name_scope('mentee_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-2, seed=seed), trainable=self.trainable,
								 name='mentee_weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
			# self.conv2_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv2_1')(self.conv2_1)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')

		with tf.name_scope('mentee_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
													 stddev=1e-2, seed=seed), trainable=self.trainable,
								 name='mentee_weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
			# self.conv3_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv3_1')(self.conv3_1)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv3_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool3')

		with tf.name_scope('mentee_fc3') as scope:
			shape = int(np.prod(self.pool3.get_shape()[1:]))
			fc3w = tf.Variable(tf.truncated_normal([shape, num_classes],
												   dtype=tf.float32, stddev=1e-2, seed=seed), trainable=self.trainable,
							   name='mentee_weights')
			fc3b = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
							   trainable=self.trainable, name='mentee_biases')
			pool1_flat = tf.reshape(self.pool3, [-1, shape])
			self.fc3l = tf.nn.bias_add(tf.matmul(pool1_flat, fc3w), fc3b)
			# self.fc3 = tf.nn.relu(fc3l)
			self.parameters += [fc3w, fc3b]

		self.softmax = tf.nn.softmax(self.fc3l / temp_softmax)
		return self

	def build_conv2fc1(self, rgb, num_classes, temp_softmax, seed,train_mode):

		print("independent student build_conv2fc1")

		K.set_learning_phase(True)
		# conv1_1
		with tf.name_scope('mentee_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, self.num_channels, 64], dtype=tf.float32,
													 stddev=1e-2, seed=seed), trainable=self.trainable,
								 name='mentee_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			# out = self.extra_regularization(out)

			self.conv1_1 = tf.nn.relu(out, name=scope)
			# self.conv1_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv1_1')(self.conv1_1)
			self.parameters += [kernel, biases]

		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')
		# conv2_1
		with tf.name_scope('mentee_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-2, seed=seed), trainable=self.trainable,
								 name='mentee_weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
			# self.conv2_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv2_1')(self.conv2_1)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')

		with tf.name_scope('mentee_fc3') as scope:
			shape = int(np.prod(self.pool2.get_shape()[1:]))
			fc3w = tf.Variable(tf.truncated_normal([shape, num_classes],
												   dtype=tf.float32, stddev=1e-2, seed=seed), trainable=self.trainable,
							   name='mentee_weights')
			fc3b = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
							   trainable=self.trainable, name='mentee_biases')
			pool1_flat = tf.reshape(self.pool2, [-1, shape])
			self.fc3l = tf.nn.bias_add(tf.matmul(pool1_flat, fc3w), fc3b)
			# self.fc3 = tf.nn.relu(fc3l)
			self.parameters += [fc3w, fc3b]

		self.softmax = tf.nn.softmax(self.fc3l / temp_softmax)
		return self

	def build_conv1fc1(self, rgb, num_classes, temp_softmax, seed, train_mode):
		K.set_learning_phase(True)

		# conv1_1
		print("build_conv1fc1")
		with tf.name_scope('mentee_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, self.num_channels, 64], dtype=tf.float32,
													 stddev=1e-2, seed=seed), trainable=self.trainable,
								 name='mentee_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			# out = self.extra_regularization(out)

			self.conv1_1 = tf.nn.relu(out, name=scope)
			# self.conv1_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv1_1')(self.conv1_1)
			self.parameters += [kernel, biases]

		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')

		with tf.name_scope('mentee_fc3') as scope:
			shape = int(np.prod(self.pool1.get_shape()[1:]))
			fc3w = tf.Variable(tf.truncated_normal([shape, num_classes],
												   dtype=tf.float32, stddev=1e-2, seed=seed), trainable=self.trainable,
							   name='mentee_weights')
			fc3b = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
							   trainable=self.trainable, name='mentee_biases')
			pool1_flat = tf.reshape(self.pool1, [-1, shape])
			self.fc3l = tf.nn.bias_add(tf.matmul(pool1_flat, fc3w), fc3b)
			# self.fc3 = tf.nn.relu(fc3l)
			self.parameters += [fc3w, fc3b]

		self.softmax = tf.nn.softmax(self.fc3l / temp_softmax)
		return self


