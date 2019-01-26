import tensorflow as tf
import numpy as np

x = tf.constant([[1., 1.], [1., 2.]])
y = tf.reduce_mean(x)

with tf.Session() as sess:

    #tf.set_random_seed(1234)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    sess.run(tf.initialize_all_variables())
    print(sess.run(y))


    coord.request_stop()
    coord.join(threads)

