import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
import h5py as h5 
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras import backend as K
VGG_MEAN = [103.939, 116.779, 123.68]

class TeacherForCifar10(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		#self.dropout = dropout
		self.parameters = []
		#self.teacher_dict = {}

	def fc_teacher(self, input, layerName, out_filter):
		with tf.name_scope(layerName):
			shape = int(np.prod(input.get_shape()[1:]))
			fc_weights = tf.Variable(tf.truncated_normal([shape, out_filter], dtype=tf.float32, stddev=1e-2),trainable=self.trainable, name='weights')
			fc_biases = tf.Variable(tf.constant(1.0, shape=[out_filter], dtype=tf.float32), trainable=self.trainable,name='biases')
			input_flat = tf.reshape(input, [-1, shape])
			fc = tf.nn.bias_add(tf.matmul(input_flat, fc_weights), fc_biases)
			relu = tf.nn.relu(fc)
			if layerName == "fc3":
				return fc
			else:
				return relu

	def build_teacher_oneConvLayer(self, input, layerName, out_filter):
		with tf.name_scope(layerName):
			num_filters_in = int(input.shape[3])
			kernel = tf.Variable(tf.truncated_normal([3, 3, num_filters_in, out_filter], dtype=tf.float32, stddev=1e-2),trainable=self.trainable, name='weights')
			biases = tf.Variable(tf.constant(0.0, shape=[out_filter], dtype=tf.float32), trainable=self.trainable, name='biases')
			conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
			out = tf.nn.bias_add(conv, biases)
			out = BatchNormalization(axis=-1, name='bn')(out)
			out = tf.nn.relu(out, name="relu")
			return out

	def build_vgg16_teacher(self, images, num_classes, temp_softmax):
		print("build_vgg16_teacher")
		K.set_learning_phase(True)
		with tf.name_scope('mentor'):
			self.conv1_1 = self.build_teacher_oneConvLayer(images, "conv1_1", 64)
			self.conv1_2 = self.build_teacher_oneConvLayer(self.conv1_1, "conv1_2", 64)
			pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			self.conv2_1 = self.build_teacher_oneConvLayer(pool1, "conv2_1", 128)
			self.conv2_2 = self.build_teacher_oneConvLayer(self.conv2_1, "conv2_2", 128)
			pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			self.conv3_1 = self.build_teacher_oneConvLayer(pool2, "conv3_1", 256)
			self.conv3_2 = self.build_teacher_oneConvLayer(self.conv3_1, "conv3_2", 256)
			self.conv3_3 = self.build_teacher_oneConvLayer(self.conv3_2, "conv3_3", 256)
			pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			self.conv4_1 = self.build_teacher_oneConvLayer(pool3, "conv4_1", 512)
			self.conv4_2 = self.build_teacher_oneConvLayer(self.conv4_1, "conv4_2", 512)
			self.conv4_3 = self.build_teacher_oneConvLayer(self.conv4_2, "conv4_3", 512)
			pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			self.conv5_1 = self.build_teacher_oneConvLayer(pool4, "conv5_1", 512)
			self.conv5_2 = self.build_teacher_oneConvLayer(self.conv5_1, "conv5_2", 512)
			self.conv5_3 = self.build_teacher_oneConvLayer(self.conv5_2, "conv5_3", 512)
			pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
			print(pool5)

			self.fc1 = self.fc_teacher(pool5, "fc1", 4096)
			self.fc2 = self.fc_teacher(self.fc1, "fc2", 4096)
			self.fc3 = self.fc_teacher(self.fc2, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(self.fc1)
			print(self.fc2)
			print(self.fc3)
			return self

	def build_vgg16_teacher_deleteFilters(self, images, num_classes, temp_softmax):
		print("build_vgg16_teacher")
		K.set_learning_phase(True)
		width = [64-10, 64-6, 128-18, 128-26, 256-52, 256-46, 256-46, 512-138, 512-134, 512-132, 512-146, 512-172, 512-258, 4096-892, 4096-2768] #60per,0.5
		#width = [64-10, 64-6, 128-18, 128-26, 256-52, 256-46, 256-46, 512-138, 512-134, 512-132, 512-146, 512-172, 512-258, 4096-1162, 4096-2984] #60per,0.5

		with tf.name_scope('mentor'):
			self.conv1_1 = self.build_teacher_oneConvLayer(images, "conv1_1", width[0])
			self.conv1_2 = self.build_teacher_oneConvLayer(self.conv1_1, "conv1_2", width[1])
			pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			self.conv2_1 = self.build_teacher_oneConvLayer(pool1, "conv2_1", width[2])
			self.conv2_2 = self.build_teacher_oneConvLayer(self.conv2_1, "conv2_2", width[3])
			pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			self.conv3_1 = self.build_teacher_oneConvLayer(pool2, "conv3_1", width[4])
			self.conv3_2 = self.build_teacher_oneConvLayer(self.conv3_1, "conv3_2", width[5])
			self.conv3_3 = self.build_teacher_oneConvLayer(self.conv3_2, "conv3_3", width[6])
			pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			self.conv4_1 = self.build_teacher_oneConvLayer(pool3, "conv4_1", width[7])
			self.conv4_2 = self.build_teacher_oneConvLayer(self.conv4_1, "conv4_2", width[8])
			self.conv4_3 = self.build_teacher_oneConvLayer(self.conv4_2, "conv4_3", width[9])
			pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			self.conv5_1 = self.build_teacher_oneConvLayer(pool4, "conv5_1", width[10])
			self.conv5_2 = self.build_teacher_oneConvLayer(self.conv5_1, "conv5_2", width[11])
			self.conv5_3 = self.build_teacher_oneConvLayer(self.conv5_2, "conv5_3", width[12])
			pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
			print(pool5)

			self.fc1 = self.fc_teacher(pool5, "fc1", 4096)
			self.fc2 = self.fc_teacher(self.fc1, "fc2", 4096)
			self.fc3 = self.fc_teacher(self.fc2, "fc3", num_classes)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(self.fc1)
			print(self.fc2)
			print(self.fc3)
			return self

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=self.fc3, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')

	def training(self, loss, learning_rate, global_step):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def _calc_num_trainable_params(self):
		self.num_trainable_params = np.sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
		print('number of trainable params: ' + str(self.num_trainable_params))