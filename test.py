import tensorflow as tf

cosine1_11 = tf.constant(5.0)
cosine1_12 = tf.constant(4.0)
#bigger = tf.greater(cosine1_11, cosine1_12)
def return_1():
    return tf.constant(1)
def return_2():
    return tf.constant(2)
cosine = tf.cond(cosine1_11 > cosine1_12, return_1, return_2)
with tf.Session() as sess:

    print sess.run([cosine1_11,cosine1_12, cosine])
    #print(sess.run(cosine1))