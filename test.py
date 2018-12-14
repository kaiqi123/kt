import tensorflow as tf

sess = tf.Session()

dataset = tf.data.Dataset.range(5)
dataset = dataset.repeat(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


sess.run(iterator.initializer)

for i in range(4):
  value = sess.run(next_element)
  print(value)



"""
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))

iterator = dataset2.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer)
print(type(dataset2))

"""