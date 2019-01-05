
"""
print("add two layers: initialization")
for var in tf.global_variables():
    if var.op.name == "mentor_conv1_1/mentor_weights":
        self.mentee_data_dict.parameters[0].assign(var.eval(session=sess)).eval(session=sess)

    if var.op.name == "mentor_conv2_1/mentor_weights":
        self.mentee_data_dict.parameters[2].assign(var.eval(session=sess)).eval(session=sess)

    if var.op.name == "mentor_conv3_1/mentor_weights":
        self.mentee_data_dict.parameters[4].assign(var.eval(session=sess)).eval(session=sess)

    if var.op.name == "mentor_conv4_1/mentor_weights":
        self.mentee_data_dict.parameters[6].assign(var.eval(session=sess)).eval(session=sess)

    if var.op.name == "mentor_conv5_1/mentor_weights":
        self.mentee_data_dict.parameters[8].assign(var.eval(session=sess)).eval(session=sess)

    if var.op.name == "mentor_fc2/mentor_weights":
        self.mentee_data_dict.parameters[14].assign(var.eval(session=sess)).eval(session=sess)

    if var.op.name == "mentor_fc3/mentor_weights":
        self.mentee_data_dict.parameters[16].assign(var.eval(session=sess)).eval(session=sess)
"""

"""
def define_outputTrain_loss_and_optimizers(self, lr):


    print("define outputTrain loss and optimizers")

    l1_var_list = []
    l2_var_list = []
    l3_var_list = []
    l4_var_list = []
    l5_var_list = []

    l1_var_list.append([var for var in tf.global_variables() if var.op.name == "mentee_conv1_1/mentee_weights"][0])
    l2_var_list.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])
    l3_var_list.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
    l4_var_list.append([var for var in tf.global_variables() if var.op.name=="mentee_conv4_1/mentee_weights"][0])
    l5_var_list.append([var for var in tf.global_variables() if var.op.name=="mentee_conv5_1/mentee_weights"][0])

    self.ph_mentor_out1 = tf.placeholder(tf.float32, shape=self.mentor_data_dict.conv1_2.shape)
    self.ph_mentor_out2 = tf.placeholder(tf.float32, shape=self.mentor_data_dict.conv2_1.shape)
    self.ph_mentor_out3 = tf.placeholder(tf.float32, shape=self.mentor_data_dict.conv3_1.shape)
    self.ph_mentor_out4 = tf.placeholder(tf.float32, shape=self.mentor_data_dict.conv4_2.shape)
    self.ph_mentor_out5 = tf.placeholder(tf.float32, shape=self.mentor_data_dict.conv5_2.shape)

    self.mentor_out1 = tf.get_variable(name="mentor_output_layer1", shape=self.mentor_data_dict.conv1_2.shape)
    self.mentor_out2 = tf.get_variable(name="mentor_output_layer2", shape=self.mentor_data_dict.conv2_1.shape)
    self.mentor_out3 = tf.get_variable(name="mentor_output_layer3", shape=self.mentor_data_dict.conv3_1.shape)
    self.mentor_out4 = tf.get_variable(name="mentor_output_layer4", shape=self.mentor_data_dict.conv4_2.shape)
    self.mentor_out5 = tf.get_variable(name="mentor_output_layer5", shape=self.mentor_data_dict.conv5_2.shape)

    self.mentor_out1.assign(self.ph_mentor_out1)
    self.mentor_out2.assign(self.ph_mentor_out2)
    self.mentor_out3.assign(self.ph_mentor_out3)
    self.mentor_out4.assign(self.ph_mentor_out4)
    self.mentor_out5.assign(self.ph_mentor_out5)

    self.l1_interval = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_out1, self.mentee_data_dict.conv1_1))))
    self.l2_interval = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_out2, self.mentee_data_dict.conv2_1))))
    self.l3_interval = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_out3, self.mentee_data_dict.conv3_1))))
    self.l4_interval = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_out4, self.mentee_data_dict.conv4_1))))
    self.l5_interval = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_out5, self.mentee_data_dict.conv5_1))))

    self.train_op1_interval = tf.train.AdamOptimizer(lr).minimize(self.l1_interval, var_list=l1_var_list)
    self.train_op2_interval = tf.train.AdamOptimizer(lr).minimize(self.l2_interval, var_list=l2_var_list)
    self.train_op3_interval = tf.train.AdamOptimizer(lr).minimize(self.l3_interval, var_list=l3_var_list)
    self.train_op4_interval = tf.train.AdamOptimizer(lr).minimize(self.l4_interval, var_list=l4_var_list)
    self.train_op5_interval = tf.train.AdamOptimizer(lr).minimize(self.l5_interval, var_list=l5_var_list)

def define_lossValueTrain_loss_and_optimizers(self, lr):

    print("define lossValueTrain loss and optimizers")
    l1_var_list = []
    l1_var_list.append([var for var in tf.global_variables() if var.op.name == "mentee_conv1_1/mentee_weights"][0])
    self.ph_loss1 = tf.placeholder(tf.float32, shape=[1])
    self.loss1 = tf.get_variable(name="loss_value1", shape=[1])
    self.loss1.assign(self.ph_loss1)
    self.train_op1_lossTrain1 = tf.train.AdamOptimizer(lr).minimize(self.loss1, var_list=l1_var_list)
    
if FLAGS.interval_output_train:
    self.define_outputTrain_loss_and_optimizers(lr)

if FLAGS.interval_lossValue_train:
    self.define_lossValueTrain_loss_and_optimizers(lr)
    
if FLAGS.interval_output_train:

    if (i % FLAGS.num_iterations == 0):

        _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
        _, self.loss_value1, mentor_out1 = sess.run([self.train_op1, self.l1, self.mentor_data_dict.conv1_2], feed_dict=feed_dict)
        _, self.loss_value2, mentor_out2 = sess.run([self.train_op2, self.l2, self.mentor_data_dict.conv2_1], feed_dict=feed_dict)
        _, self.loss_value3, mentor_out3 = sess.run([self.train_op3, self.l3, self.mentor_data_dict.conv3_1], feed_dict=feed_dict)
        _, self.loss_value4, mentor_out4 = sess.run([self.train_op4, self.l4, self.mentor_data_dict.conv4_2], feed_dict=feed_dict)
        _, self.loss_value5, mentor_out5 = sess.run([self.train_op5, self.l5, self.mentor_data_dict.conv5_2], feed_dict=feed_dict)

        sess.run(self.mentor_out1, feed_dict = {self.ph_mentor_out1: mentor_out1})
        sess.run(self.mentor_out2, feed_dict = {self.ph_mentor_out2: mentor_out2})
        sess.run(self.mentor_out3, feed_dict = {self.ph_mentor_out3: mentor_out3})
        sess.run(self.mentor_out4, feed_dict = {self.ph_mentor_out4: mentor_out4})
        sess.run(self.mentor_out5, feed_dict = {self.ph_mentor_out5: mentor_out5})

    else:
        _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
        _, self.loss_value1 = sess.run([self.train_op1_interval, self.l1], feed_dict=feed_dict)
        _, self.loss_value2 = sess.run([self.train_op2_interval, self.l2], feed_dict=feed_dict)
        _, self.loss_value3 = sess.run([self.train_op3_interval, self.l3], feed_dict=feed_dict)
        _, self.loss_value4 = sess.run([self.train_op4_interval, self.l4], feed_dict=feed_dict)
        _, self.loss_value5 = sess.run([self.train_op5_interval, self.l5], feed_dict=feed_dict)

elif FLAGS.interval_lossValue_train:

    if (i % FLAGS.num_iterations == 0):
        print(i)
        _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
        sess.run(self.loss1, feed_dict={self.ph_loss1: [self.loss_value1]})

        _, self.loss_value1 = sess.run([self.train_op1_lossTrain1, self.l1], feed_dict=feed_dict)
"""
