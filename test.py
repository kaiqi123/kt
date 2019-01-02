import tensorflow as tf
import numpy as np

dataset_path = "./"
train_labels_file = "caltech101-train.txt"
filename_queue = tf.train.string_input_producer([dataset_path + train_labels_file], num_epochs=None)
reader = tf.TextLineReader()
key_temp, value_temp = reader.read(filename_queue)