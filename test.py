import tensorflow as tf

cosine1_11 = tf.constant(3.0)
cosine1_12 = tf.constant(4.0)
bigger = tf.greater(cosine1_11, cosine1_12)
c = tf.cond(cosine1_11 > cosine1_12, lamba: cosine1_11, lamba: cosine1_11)
if temp:
    print("1")
    cosine1 = cosine1_11
else:
    print("2")
    cosine1 = cosine1_12

with tf.Session() as sess:

    print sess.run([cosine1_11,cosine1_12, temp, cosine1])
    #print(sess.run(cosine1))