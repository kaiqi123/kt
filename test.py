import tensorflow as tf


def find_largest_cosine_among_three_numbers(num1, num2, num3):
    def return_1():
        return tf.constant(1)

    def return_2():
        return tf.constant(2)

    def return_3():
        return tf.constant(3)

    def compare_number13(num1, num3):
        result = tf.cond(num1 > num3, return_1, return_3)
        return result

    def compare_two_number23(num2, num3):
        result = tf.cond(num2 > num3, return_2, return_3)
        return result

    result = tf.cond(num1>num2, lambda: compare_number13(num1, num3), lambda: compare_two_number23(num2, num3))

    return result

x = tf.constant(8.0)
y = tf.constant(9.0)
z = tf.constant(4.0)

cosine = find_largest_cosine_among_three_numbers(x,y,z)
#value,index = tf.nn.top_k([x,y,z])
#tf.cond(index==[0], return_1, re)

with tf.Session() as sess:

    print sess.run([x, y, z, cosine])
    #print(sess.run(cosine1))