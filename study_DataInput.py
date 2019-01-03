import tensorflow as tf
import numpy as np
from DataInput import DataInput


dataset_path = "./"
test_dataset = "dataset_input/temp.txt"
batch_size = 3
num_testing_examples = 1829
image_width = 224
image_height = 224
num_channels = 3
seed = 1234
dataset = 'caltech101'

"""
data_input_test = DataInput(dataset_path, test_dataset, batch_size, num_testing_examples,
                                        image_width, image_height, num_channels, seed, dataset)

"""
filename_queue = tf.train.string_input_producer([dataset_path + test_dataset], num_epochs=None)
reader = tf.TextLineReader()
key_temp, value_temp = reader.read(filename_queue)
record_defaults = [[1], ['']]
col1, col2 = tf.decode_csv(value_temp, record_defaults=record_defaults)

file_content = tf.read_file(col2)

if dataset == 'caltech101':
    train_image = tf.image.decode_jpeg(file_content, channels=num_channels)
    distorted_image = tf.image.random_flip_left_right(train_image)
    train_image = tf.image.per_image_standardization(distorted_image)
    train_image = tf.image.resize_images(train_image, [image_width, image_height])

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch = tf.train.shuffle_batch (
        [train_image, col1], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, seed=seed)


with tf.Session() as sess:

    #tf.set_random_seed(1234)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    sess.run(tf.initialize_all_variables())

    for i in range(1):
        #images_feed, labels_feed = sess.run([data_input_test.example_batch, data_input_test.label_batch])
        images_feed, labels_feed = sess.run([example_batch, label_batch])
        print(images_feed.shape)
        print(labels_feed)

        images_feed, labels_feed = sess.run([example_batch, label_batch])
        print(images_feed.shape)
        print(labels_feed)

    coord.request_stop()
    coord.join(threads)
