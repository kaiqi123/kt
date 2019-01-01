import tensorflow as tf

"""
sess = tf.Session()

dataset = tf.data.Dataset.range(5)
dataset = dataset.repeat(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


sess.run(iterator.initializer)

for i in range(4):
  value = sess.run(next_element)
  print(value)


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))

iterator = dataset2.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer)
print(type(dataset2))



        dataset_mentee = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder, phase_train))
        iterator_mentee = dataset_mentee.make_initializable_iterator()
        images_place, labels_placeholder, phase_train = iterator_mentee.get_next()

        self.l1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv1_2, self.mentee_data_dict.conv1_1))))
        l1_var_list = []
        l1_var_list.append([var for var in tf.global_variables() if var.op.name == "mentee_conv1_1/mentee_weights"][0])
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(self.l1, var_list=l1_var_list)



import tensorflow as tf
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create graph
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = tf.add(a, b, name="addition")
# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# launch the graph in a session
with tf.Session() as sess:
    # or creating the writer inside the session
    writer = tf.summary.FileWriter('tensorboard/', sess.graph)
    print(sess.run(c))
"""
import tensorflow as tf
import numpy as np

a = tf.get_variable("L_enc", shape = [1])
ph = tf.placeholder(tf.float32, shape = [1])
a.assign(ph)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  embedding = 2.0
  print(sess.run(a, {ph: embedding}))
