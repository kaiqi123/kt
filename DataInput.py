import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random

class DataInput(object):

	def __init__(self, dataset_path, train_labels_file, batch_size, num_examples, image_width, image_height, num_channels, seed, dataset):
		self.dataset_path = dataset_path
		self.train_labels_file = train_labels_file
		self.num_examples = num_examples
		self.image_height = image_height
		self.image_width = image_width
		self.num_channels = num_channels
		self.seed = seed
		self.dataset = dataset

		self.check_data(train_labels_file)
		# Create the File Name queue
		self.filename_queue = tf.train.string_input_producer([self.dataset_path + self.train_labels_file], num_epochs=None)
		# Reading the file line by line
		self.reader = tf.TextLineReader()
		# Parse the line of CSV
		self.key_temp, self.value_temp =  self.reader.read(self.filename_queue)
		self.record_defaults = [[1], ['']]
		self.col1, self.col2 = tf.decode_csv(self.value_temp, record_defaults=self.record_defaults)
    
		# Decode the data into JPEG
		self.decode_jpeg()

		# setup the input pipeline
		self.input_pipeline(batch_size)

	def input_pipeline(self, batch_size,num_epochs=None):
		self.min_after_dequeue = 10000
		self.capacity = self.min_after_dequeue + 3 * batch_size
		self.example_batch, self.label_batch = tf.train.shuffle_batch (
		    	[self.train_image, self.col1], batch_size=batch_size, capacity=self.capacity,
             		min_after_dequeue=self.min_after_dequeue, seed=self.seed)
		return self.example_batch, self.label_batch

	def decode_jpeg(self):
		file_content = tf.read_file(self.col2)
		self.train_image = tf.image.decode_png(file_content, channels=self.num_channels)
		distorted_image = tf.image.random_flip_left_right(self.train_image)
		self.train_image = tf.image.per_image_standardization(distorted_image)
		self.train_image = tf.image.resize_images(self.train_image, [self.image_width, self.image_height])

	def check_data(self, train_labels_file):
		f = open(train_labels_file)
		lines = f.readlines()
		f.close()
		labels = []
		for line in lines:
			label = int(line.split(",")[0])
			if  label not in labels:
				labels.append(label)
		print("Read data from file: "+str(train_labels_file))
		print("The number of data is: " + str(len(lines)))
		print("The labels are: " + str(sorted(labels)))


