import tensorflow as tf

tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3])
v2 = tf.get_variable("v2", [5])

#print("v1 : %s" % v1.eval())
#print("v2 : %s" % v2.eval())

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  #sess.run(tf.global_variables_initializer())
  #saver.save(sess, "./temp/studySaveRestore")
  saver.restore(sess, "./temp/studySaveRestore")

  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
