
"""
    elif FLAGS.interval_lossValue_train:

        if (i % FLAGS.num_iterations == 0):
            print(i)
            _, self.loss_value1= sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            sess.run(self.loss1, feed_dict={self.ph_loss1: [self.loss_value1]})
        else:
            _, self.loss_value1 = sess.run([self.train_op1_lossTrain1, self.l1], feed_dict=feed_dict)
                    
                    
def define_lossValueTrain_loss_and_optimizers(self, lr):

        print("define lossValueTrain loss and optimizers")
        l1_var_list = []
        l1_var_list.append([var for var in tf.global_variables() if var.op.name == "mentee_conv1_1/mentee_weights"][0])
        self.ph_loss1 = tf.placeholder(tf.float32, shape=[1])
        self.loss1 = tf.get_variable(name="loss_value1", shape=[1])
        self.loss1.assign(self.ph_loss1)
        self.train_op1_lossTrain1 = tf.train.AdamOptimizer(lr).minimize(self.loss1, var_list=l1_var_list)
        
        
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

