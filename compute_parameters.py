import tensorflow as tf
import numpy as np


input = 32*32*3
pool1 = 16*16*64
pool2 = 8*8*128
pool3 = 4*4*256
pool4 = 2*2*512
pool5 = 1*1*512

fc1 = pool5*4096
fc2 = 4096*4096
fc3 = 4096*10
fc_total = fc1 + fc2 +fc3


m = np.power(10,6)
VGG16 = 138*m - 4096000 - 16777216 - 102760448 + fc_total
print(VGG16) #33M


conv11 = (3*3*3)*50

input = 32*32*3
pool1 = 16*16*50
fc3 = pool1*200
softmax = 200*10
conv2_fc1 = conv11 + fc3
print(conv2_fc1) #157376

#conv21 = (3*3*64)*128
#fc3 = pool2*10
#conv2_fc1 = conv11 + conv21 + fc3
#print(conv2_fc1) #157376

"""
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
"""

