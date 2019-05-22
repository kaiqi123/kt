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
		self.dropout = dropout
		self.parameters = []

	def fc_teacher(input, layerName, out_filter, is_training):
		with tf.name_scope(layerName):
			if layerName == "fc1":
				shape = int(np.prod(input.get_shape()[1:]))
				fc_weights = tf.Variable(tf.truncated_normal([shape, out_filter], dtype=tf.float32, stddev=1e-2),trainable=is_training, name='weights')
				fc_biases = tf.Variable(tf.constant(1.0, shape=[out_filter], dtype=tf.float32), trainable=is_training,name='biases')
				input_flat = tf.reshape(input, [-1, shape])
				fc = tf.nn.bias_add(tf.matmul(input_flat, fc_weights), fc_biases)
				fc = tf.nn.relu(fc)
				fc = BatchNormalization(axis=-1, name=layerName + 'bn')(fc)
			elif layerName == "fc2":
				fc_weights = tf.Variable(tf.truncated_normal([4096, out_filter], dtype=tf.float32, stddev=1e-2),trainable=is_training, name='weights')
				fc_biases = tf.Variable(tf.constant(1.0, shape=[out_filter], dtype=tf.float32), trainable=is_training,name='biases')
				fc = tf.nn.bias_add(tf.matmul(input, fc_weights), fc_biases)
				fc = tf.nn.relu(fc)
				fc = BatchNormalization(axis=-1, name=layerName + 'bn')(fc)
				if is_training == True:
					fc = tf.nn.dropout(fc, 0.5)
			elif layerName == "fc3":
				fc_weights = tf.Variable(tf.truncated_normal([4096, out_filter], dtype=tf.float32, stddev=1e-2),trainable=is_training, name='weights')
				fc_biases = tf.Variable(tf.constant(1.0, shape=[out_filter], dtype=tf.float32), trainable=is_training,name='biases')
				fc = tf.nn.bias_add(tf.matmul(input, fc_weights), fc_biases)
			return fc

	def build_teacher_oneConvLayer(input, layerName, out_filter, is_training):
		with tf.name_scope(layerName):
			num_filters_in = int(input.shape[3])
			kernel = tf.Variable(tf.truncated_normal([3, 3, num_filters_in, out_filter], dtype=tf.float32, stddev=1e-2),trainable=is_training, name='weights')
			conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
			relu = tf.nn.relu(conv, name="relu")
			bn = BatchNormalization(axis=-1, name='bn')(relu)
			return bn

	def build_vgg16_teacher(self, images, num_classes, temp_softmax, is_training):
	#def build_vgg16_teacher(images, num_classes, is_training):
		K.set_learning_phase(True)
		with tf.name_scope('mentor'):
			conv1_1 = self.build_teacher_oneConvLayer(images, "conv1_1", 64, is_training)
			conv1_2 = self.build_teacher_oneConvLayer(conv1_1, "conv1_2", 64, is_training)
			pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			print(pool1)

			conv2_1 = self.build_teacher_oneConvLayer(pool1, "conv2_1", 128, is_training)
			conv2_2 = self.build_teacher_oneConvLayer(conv2_1, "conv2_2", 128, is_training)
			pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
			print(pool2)

			conv3_1 = self.build_teacher_oneConvLayer(pool2, "conv3_1", 256, is_training)
			conv3_2 = self.build_teacher_oneConvLayer(conv3_1, "conv3_2", 256, is_training)
			conv3_3 = self.build_teacher_oneConvLayer(conv3_2, "conv3_3", 256, is_training)
			pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
			print(pool3)

			conv4_1 = self.build_teacher_oneConvLayer(pool3, "conv4_1", 512, is_training)
			conv4_2 = self.build_teacher_oneConvLayer(conv4_1, "conv4_2", 512, is_training)
			conv4_3 = self.build_teacher_oneConvLayer(conv4_2, "conv4_3", 512, is_training)
			pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
			print(pool4)

			conv5_1 = self.build_teacher_oneConvLayer(pool4, "conv5_1", 512, is_training)
			conv5_2 = self.build_teacher_oneConvLayer(conv5_1, "conv5_2", 512, is_training)
			conv5_3 = self.build_teacher_oneConvLayer(conv5_2, "conv5_3", 512, is_training)
			pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
			print(pool5)

			fc1 = self.fc_teacher(pool5, "fc1", 4096, is_training)
			fc2 = self.fc_teacher(fc1, "fc2", 4096, is_training)
			self.fc3 = self.fc_teacher(fc2, "fc3", num_classes, is_training)
			self.softmax = tf.nn.softmax(self.fc3 / temp_softmax)
			print(fc1)
			print(fc2)
			print(self.fc3)
			return self

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=self.fc3l, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')

	def training(self, loss, learning_rate, global_step):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op
