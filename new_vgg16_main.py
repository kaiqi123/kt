import argparse
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from DataInput import DataInput
from teacherCifar10 import TeacherForCifar10
from teacherCaltech101 import MentorForCaltech101
from studentModels import Mentee

dataset_path = "./"
tf.reset_default_graph()
NUM_ITERATIONS = 19500  #ciafr10: 50000/128=390, 390*50=19500; caltech101: 234*50=11700
#NUM_ITERATIONS = 702   # 234*3=702
SUMMARY_LOG_DIR="./summary-log"
LEARNING_RATE_DECAY_FACTOR = 0.9809
NUM_EPOCHS_PER_DECAY = 1.0
validation_accuracy_list = []
test_accuracy_list = []
seed = 1234
alpha = 0.2
random_count = 0
count_cosine = [0,0,0,0,0,0,0,0,0,0,0,0,0]
teacher_alltrue_list = []
teacher_alltrue_list_127 = []
teacher_alltrue_list_126 = []

class VGG16(object):

    ### placeholders are filled with actual images and labels which are fed to the network while training.
    def fill_feed_dict(self, data_input, images_pl, labels_pl, sess):
        images_feed, labels_feed = sess.run([data_input.example_batch, data_input.label_batch])
        feed_dict = {images_pl: images_feed, labels_pl: labels_feed}
        return feed_dict, images_feed, labels_feed

    def evaluation(self, logits, labels):
        if FLAGS.top_1_accuracy:
            correct = tf.nn.in_top_k(logits, labels, 1)
        elif FLAGS.top_3_accuracy:
            correct = tf.nn.in_top_k(logits, labels, 3)
        elif FLAGS.top_5_accuracy:
            print("top_5_accuracy")
            correct = tf.nn.in_top_k(logits, labels, 5)
        else:
            raise ValueError("Not found top_1&3&5_accuracy")
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def do_eval(self, sess, eval_correct, logits, images_placeholder, labels_placeholder, dataset, mode):
        if mode == 'Test':
            steps_per_epoch = FLAGS.num_testing_examples //FLAGS.batch_size
            num_examples = steps_per_epoch * FLAGS.batch_size
        if mode == 'Train':
            steps_per_epoch = FLAGS.num_training_examples //FLAGS.batch_size
            num_examples = steps_per_epoch * FLAGS.batch_size
        if mode == 'Validation':
            steps_per_epoch = FLAGS.num_validation_examples //FLAGS.batch_size
            num_examples = steps_per_epoch * FLAGS.batch_size

        true_count = 0
        for step in xrange(steps_per_epoch):
            if FLAGS.dataset == 'mnist':
                feed_dict = {images_placeholder: np.reshape(dataset.test.next_batch(FLAGS.batch_size)[0], [FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels]), labels_placeholder: dataset.test.next_batch(FLAGS.batch_size)[1]}
            else:
                feed_dict, images_feed, labels_feed = self.fill_feed_dict(dataset, images_placeholder, labels_placeholder, sess)
            count = sess.run(eval_correct, feed_dict=feed_dict)
            true_count = true_count + count

        precision = float(true_count) / num_examples
        print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %(num_examples, true_count, precision))

        if mode == 'Test':
            test_accuracy_list.append(precision)

    def define_independent_student(self, images_placeholder, labels_placeholder, seed, global_step, sess):
        print("Build Independent student")
        student = Mentee(seed)

        if FLAGS.num_optimizers == 5:
            mentee_data_dict = student.build_student_conv5fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        else:
            raise ValueError("Not found num_optimizers")

        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        self.lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,staircase=True)

        self.loss = student.loss(labels_placeholder)
        self.softmax = mentee_data_dict.softmax
        self.train_op = student.training(self.loss, self.lr, global_step)

        """
        # DeCAF phase2
        fc_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/fc3/weights"
                       or var.op.name == "mentee/fc3/biases"]
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=global_step, var_list=fc_var_list)
        print("fc_var_list is: "+str(fc_var_list))
        """

        for tvar in tf.trainable_variables():
            print(tvar)
        print('Mentee, trainable variables: %d' % len(tf.trainable_variables()))

        student._calc_num_trainable_params()
        init = tf.initialize_all_variables()
        sess.run(init)
        self.saver = tf.train.Saver()

        # DeCAF phase2, restore all weights
        #saverDeCAF = tf.train.Saver(tf.trainable_variables())
        #saverDeCAF.restore(sess, FLAGS.student_filename)


    def define_teacher(self, images_placeholder, labels_placeholder, global_step, sess):
        if FLAGS.dataset == 'cifar10':
            print("Build Teacher (cifar10)")
            mentor = TeacherForCifar10()
        elif FLAGS.dataset == 'caltech101':
            print("Build Teacher (caltech101)")
            mentor = MentorForCaltech101()
        else:
            raise ValueError("Not found dataset name")

        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        self.lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,staircase=True)

        mentor_data_dict = mentor.build_vgg16_teacher(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        #mentor_data_dict = mentor.build_vgg16_teacher_deleteFilters(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)

        self.loss = mentor.loss(labels_placeholder)

        if FLAGS.dataset == 'caltech101':
            def get_mentor_variables_to_restore():
                return [var for var in tf.global_variables() if var.op.name.startswith("mentor") and
                        (var.op.name.endswith("biases") or var.op.name.endswith("weights"))
                        and (var.op.name != ("mentor/fc3/weights")
                             and var.op.name != ("mentor/fc3/biases"))]
            variables_to_restore = get_mentor_variables_to_restore()
            for var in variables_to_restore:
                print(var)
            print("num of variables_to_restore: ", len(variables_to_restore))
            self.train_op = mentor.training(self.loss, FLAGS.learning_rate_pretrained, self.lr, global_step, variables_to_restore, mentor.get_training_vars())
        elif FLAGS.dataset == 'cifar10':
            self.train_op = mentor.training(self.loss, self.lr, global_step)
        else:
            raise ValueError("Not found dataset name")

        for tvar in tf.trainable_variables():
            print(tvar)
        print('Mentor, trainable variables: %d' % len(tf.trainable_variables()))

        mentor._calc_num_trainable_params()
        self.softmax = mentor_data_dict.softmax
        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver = tf.train.Saver()

    def initilize(self, sess):
        for var in tf.global_variables():
            if var.op.name == "mentor/conv1_1/weights":
                print("initialization: conv1_1 weights")
                self.mentee_data_dict.parameters[0].assign(var.eval(session=sess)).eval(session=sess)
            if var.op.name == "mentor/conv1_1/biases":
                print("initialization: conv1_1 biases")
                self.mentee_data_dict.parameters[1].assign(var.eval(session=sess)).eval(session=sess)

            if FLAGS.num_optimizers >= 2:
                if var.op.name == "mentor/conv2_1/weights":
                    print("initialization: conv2_1 weights")
                    self.mentee_data_dict.parameters[2].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/conv2_1/biases":
                    print("initialization: conv2_1 biases")
                    self.mentee_data_dict.parameters[3].assign(var.eval(session=sess)).eval(session=sess)

            if FLAGS.num_optimizers >= 3:
                if var.op.name == "mentor/conv3_1/weights":
                    print("initialization: conv3_1 weights")
                    self.mentee_data_dict.parameters[4].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/conv3_1/biases":
                    print("initialization: conv3_1 biases")
                    self.mentee_data_dict.parameters[5].assign(var.eval(session=sess)).eval(session=sess)

            if FLAGS.num_optimizers >= 4:
                if var.op.name == "mentor/conv4_1/weights":
                    print("initialization: conv4_1 weights")
                    self.mentee_data_dict.parameters[6].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/conv4_1/biases":
                    print("initialization: conv4_1 biases")
                    self.mentee_data_dict.parameters[7].assign(var.eval(session=sess)).eval(session=sess)

            if FLAGS.num_optimizers == 5:
                if var.op.name == "mentor/conv5_1/weights":
                    print("initialization: conv5_1 weights")
                    self.mentee_data_dict.parameters[8].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/conv5_1/biases":
                    print("initialization: conv5_1 biases")
                    self.mentee_data_dict.parameters[9].assign(var.eval(session=sess)).eval(session=sess)

                #if var.op.name == "mentor/fc1/weights":
                #    print("initialization: fc1 weights")
                #    self.mentee_data_dict.parameters[10].assign(var.eval(session=sess)).eval(session=sess)
                #if var.op.name == "mentor/fc1/biases":
                #    print("initialization: fc1 biases")
                #    self.mentee_data_dict.parameters[11].assign(var.eval(session=sess)).eval(session=sess)

            if FLAGS.num_optimizers == 6:
                if var.op.name == "mentor/conv5_1/weights":
                    print("initialization: conv5_1 weights")
                    self.mentee_data_dict.parameters[8].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/conv5_1/biases":
                    print("initialization: conv5_1 biases")
                    self.mentee_data_dict.parameters[9].assign(var.eval(session=sess)).eval(session=sess)

                if var.op.name == "mentor/fc2/weights":
                    print("initialization: fc2 weights")
                    self.mentee_data_dict.parameters[14].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/fc2/biases":
                    print("initialization: fc2 biases")
                    self.mentee_data_dict.parameters[15].assign(var.eval(session=sess)).eval(session=sess)

    def caculate_rmse_loss(self):

        def zero_pad(inputs, in_filter, out_filter):
            outputs = tf.pad(inputs,[[0, 0], [0, 0], [0, 0], [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            return outputs

        def build_loss(teacher_layer, student_layer):
            if student_layer.get_shape()[3] != teacher_layer.get_shape()[3]:
                tf.logging.info("Zero padding on student: {}".format(student_layer))
                student_layer = zero_pad(student_layer, student_layer.get_shape()[3], teacher_layer.get_shape()[3])
            loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_layer, student_layer))))
            return loss_layer

        self.l1 = build_loss(self.mentor_data_dict.conv1_2, self.mentee_data_dict.conv1_1)
        self.l2 = build_loss(self.mentor_data_dict.conv2_2, self.mentee_data_dict.conv2_1)
        self.l3 = build_loss(self.mentor_data_dict.conv3_3, self.mentee_data_dict.conv3_1)
        self.l4 = build_loss(self.mentor_data_dict.conv4_3, self.mentee_data_dict.conv4_1)
        self.l5 = build_loss(self.mentor_data_dict.conv5_3, self.mentee_data_dict.conv5_1)
        self.loss_fc3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.fc3, self.mentee_data_dict.fc3))))
        self.loss_softmax = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.softmax, self.mentee_data_dict.softmax))))

        #self.loss_list = [self.l1,self.l2,self.l3,self.l4,self.l5, self.loss]
        #self.loss_list = [self.l1,self.l2, self.l3, self.loss]
        #self.loss_list = [self.loss_softmax]
        #self.loss_list = [self.loss_softmax, self.loss]
        self.loss_list = [self.l1,self.l2,self.loss_fc3]
        print("Number of loss is: "+str(len(self.loss_list)))


    def define_multiple_optimizers(self, lr, global_step): #Note global_step!!!!
        print("define multiple optimizers.")
        tvars = [var for var in tf.trainable_variables() if var.op.name.startswith("mentee")]
        self.train_op_fc3 = tf.train.AdamOptimizer(lr).minimize(self.loss_fc3, var_list=tvars, global_step=global_step)
        self.train_op_softmax = tf.train.AdamOptimizer(lr).minimize(self.loss_softmax, var_list=tvars)
        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=tvars)
        for var in tvars:
            print(var)
        print('num of mentee trainable_variables: %d' % len(tvars))

        l1_var_list = [var for var in tf.trainable_variables() if var.op.name == "mentee/conv1_1/weights"
                       or var.op.name == "mentee/conv1_1/biases"]
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(self.l1, var_list=l1_var_list)
        print(l1_var_list)

        l2_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/conv2_1/weights"
                       or var.op.name == "mentee/conv2_1/biases"]
        self.train_op2 = tf.train.AdamOptimizer(lr).minimize(self.l2, var_list=l2_var_list)
        print(l2_var_list)

        l3_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/conv3_1/weights"
                       or var.op.name == "mentee/conv3_1/biases"]
        self.train_op3 = tf.train.AdamOptimizer(lr).minimize(self.l3, var_list=l3_var_list)
        print(l3_var_list)

        l4_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/conv4_1/weights"
                       or var.op.name == "mentee/conv4_1/biases"]
        self.train_op4 = tf.train.AdamOptimizer(lr).minimize(self.l4, var_list=l4_var_list)
        print(l4_var_list)

        l5_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/conv5_1/weights"
                       or var.op.name == "mentee/conv5_1/biases"]
        self.train_op5 = tf.train.AdamOptimizer(lr).minimize(self.l5, var_list=l5_var_list)
        print(l5_var_list)

        #self.train_op_list = [self.train_op_softmax]
        #self.train_op_list = [self.train_op_softmax, self.train_op0]
        #self.train_op_list = [self.train_op1, self.train_op2, self.train_op3, self.train_op4, self.train_op5, self.train_op_softmax, self.train_op0]
        #self.train_op_list = [self.train_op1, self.train_op2, self.train_op3, self.train_op4, self.train_op5, self.train_op0]
        self.train_op_list = [self.train_op1, self.train_op2, self.train_op_fc3]
        print("Number of optimizers is: "+str(len(self.train_op_list)))

    def get_variables_for_fitnet_phase1(self):
        l1_var_list = [var for var in tf.trainable_variables() if var.op.name == "mentee/conv1_1/weights" or var.op.name == "mentee/conv1_1/biases"]
        l2_var_list = [var for var in tf.trainable_variables() if var.op.name == "mentee/conv2_1/weights" or var.op.name == "mentee/conv2_1/biases"]
        l3_var_list = [var for var in tf.trainable_variables() if var.op.name == "mentee/conv3_1/weights" or var.op.name == "mentee/conv3_1/biases"]
        variables_for_fitnet_phase1 = l1_var_list + l2_var_list + l3_var_list
        return variables_for_fitnet_phase1

    def build_optimizer_fitnet_phase1(self, lr):
        print("build_optimizer_fitnet_phase1")
        def zero_pad(inputs, in_filter, out_filter):
            outputs = tf.pad(inputs,[[0, 0], [0, 0], [0, 0], [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            return outputs

        def build_loss(teacher_layer, student_layer):
            if student_layer.get_shape()[3] != teacher_layer.get_shape()[3]:
                tf.logging.info("Zero padding on student: {}".format(student_layer))
                student_layer = zero_pad(student_layer, student_layer.get_shape()[3], teacher_layer.get_shape()[3])
            loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_layer, student_layer))))
            return loss_layer

        # phase 1
        self.loss_fitnet_phase1 = build_loss(self.mentor_data_dict.conv3_3, self.mentee_data_dict.conv3_1)
        variables_for_fitnet_phase1 = self.get_variables_for_fitnet_phase1()
        self.train_op_fitnet_phase1 = tf.train.AdamOptimizer(lr).minimize(self.loss_fitnet_phase1,var_list=variables_for_fitnet_phase1)
        self.train_op_list = [self.train_op_fitnet_phase1]
        self.loss_list = [self.loss_fitnet_phase1]
        print(variables_for_fitnet_phase1)
        print("Number of optimizers is: "+str(len(self.train_op_list)))

    def build_optimizer_fitnet_phase2(self, lr):
        print("build_optimizer_fitnet_phase2")

        # ragini's code
        #alpha = 0.2
        #self.loss_softmax = (tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.softmax, self.mentee_data_dict.softmax))))
        #self.loss_fitnet_phase2 = alpha * self.loss + self.loss_softmax


        # fitnet paper
        self.lamma_KD = tf.Variable(4.0, name='lamma_KD', trainable=False)
        t = 3.0
        teacher_softmax = tf.nn.softmax(self.mentor_data_dict.fc3 / t)
        student_softmax = tf.nn.softmax(self.mentee_data_dict.fc3 / t)
        loss_softmax = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_softmax, student_softmax))))
        self.loss_fitnet_phase2 = self.loss + self.lamma_KD * loss_softmax

        self.train_op_fitNet_phase2 = tf.train.AdamOptimizer(lr).minimize(self.loss_fitnet_phase2)
        self.train_op_list = [self.train_op_fitNet_phase2]
        self.loss_list = [self.loss_fitnet_phase2]
        print("Number of optimizers is: " + str(len(self.train_op_list)))


    def define_dependent_student(self, images_placeholder, labels_placeholder, seed, global_step, sess):
        if FLAGS.dataset == 'cifar10':
            print("Build teacher of dependent student (cifar10)")
            vgg16_mentor = TeacherForCifar10(False)
        elif FLAGS.dataset == 'caltech101':
            print("Build teacher of dependent student (caltech101)")
            vgg16_mentor = MentorForCaltech101(False)
        else:
            raise ValueError("Not found dataset name")

        vgg16_mentee = Mentee(seed)
        self.mentor_data_dict = vgg16_mentor.build_vgg16_teacher(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)

        if FLAGS.num_optimizers == 5:
            self.mentee_data_dict = vgg16_mentee.build_student_conv5fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        else:
            raise ValueError("Not found num_optimizers")

        self.softmax = self.mentee_data_dict.softmax
        mentor_variables_to_restore = [var for var in tf.global_variables() if var.op.name.startswith("mentor")]
        for var in mentor_variables_to_restore:
            print(var)
        print('num of mentor_variables_to_restore: %d' % len(mentor_variables_to_restore))
        print('num of mentee_variables: %d' % len([var for var in tf.global_variables() if var.op.name.startswith("mentee")]))
        print('num of global_variables: %d' % len(tf.global_variables()))

        self.loss = vgg16_mentee.loss(labels_placeholder)
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        self.lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

        if FLAGS.proposed_method:
            self.caculate_rmse_loss()
            self.define_multiple_optimizers(self.lr, global_step)
        elif FLAGS.fitnet_phase1:
            self.build_optimizer_fitnet_phase1(self.lr)
        elif FLAGS.fitnet_phase2:
            self.build_optimizer_fitnet_phase2(self.lr)
        else:
            raise ValueError("Not found method")

        vgg16_mentee._calc_num_trainable_params()
        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver(mentor_variables_to_restore)
        saver.restore(sess, FLAGS.teacher_weights_filename)

        # fitnet, phase 2
        if FLAGS.fitnet_phase2:
            print("fitnet phase2, restore variables")
            print("restore from"+str(FLAGS.fitnet_phase1_filename))
            variables_for_fitnet_phase2 = self.get_variables_for_fitnet_phase1()
            saver_fitnet_phase2 = tf.train.Saver(variables_for_fitnet_phase2)
            saver_fitnet_phase2.restore(sess, FLAGS.fitnet_phase1_filename)
            print(variables_for_fitnet_phase2)

        #if FLAGS.initialization:
        #    self.initilize(sess)

    def count_filter0_num(self, output, name):
        print(name)
        filter_count = []
        for i in range(output.shape[0]):
            # print("image: "+str(i))
            img = output[i]
            img = img.transpose(2, 0, 1)
            count = 0
            for j in range(img.shape[0]):
                """
                #count number filters whose output_wrn are all 0
                sum_oneFilter = np.sum(img[j])
                if sum_oneFilter == 0:
                    count = count + 1
                """
                # count number filters whose 90% output_wrn are 0
                num_sum = img[j].shape[0] * img[j].shape[1]
                count0_perFIlter = (num_sum - np.count_nonzero(img[j])) / num_sum
                if count0_perFIlter > 0.5:
                    count = count + 1

            filter_count.append(count)
        print(filter_count)

    def compute_0filter_of_teacherOutput(self, sess, feed_dict, images_feed, iteration):
        mentor_conv1_1, mentor_conv1_2, \
        mentor_conv2_1, mentor_conv2_2, \
        mentor_conv3_1, mentor_conv3_2, mentor_conv3_3, \
        mentor_conv4_1, mentor_conv4_2, mentor_conv4_3, \
        mentor_conv5_1, mentor_conv5_2, mentor_conv5_3, \
        mentor_fc1, mentor_fc2, mentor_fc3 \
            = sess.run([self.mentor_data_dict.conv1_1, self.mentor_data_dict.conv1_2,
                        self.mentor_data_dict.conv2_1, self.mentor_data_dict.conv2_2,
                        self.mentor_data_dict.conv3_1, self.mentor_data_dict.conv3_2, self.mentor_data_dict.conv3_3,
                        self.mentor_data_dict.conv4_1, self.mentor_data_dict.conv4_2, self.mentor_data_dict.conv4_3,
                        self.mentor_data_dict.conv5_1, self.mentor_data_dict.conv5_2, self.mentor_data_dict.conv5_3,
                        self.mentor_data_dict.fc1, self.mentor_data_dict.fc2, self.mentor_data_dict.fc3],
                       feed_dict=feed_dict)
        """
        self.count_filter0_num(mentor_conv1_1, "conv1_1")
        self.count_filter0_num(mentor_conv2_1, "conv2_1")
        self.count_filter0_num(mentor_conv2_2, "conv2_2")
        self.count_filter0_num(mentor_conv3_1, "conv3_1")
        self.count_filter0_num(mentor_conv3_2, "conv3_2")
        self.count_filter0_num(mentor_conv3_3, "conv3_3")
        self.count_filter0_num(mentor_conv4_1, "conv4_1")
        self.count_filter0_num(mentor_conv4_2, "conv4_2")
        self.count_filter0_num(mentor_conv4_3, "conv4_3")
        self.count_filter0_num(mentor_conv5_1, "conv5_1")
        self.count_filter0_num(mentor_conv5_2, "conv5_2")
        self.count_filter0_num(mentor_conv5_3, "conv5_3")
        """

        print(images_feed.shape)
        np.save("output_vgg16/filters_npy/images_feed_"+str(iteration)+".npy", images_feed)
        np.save("output_vgg16/filters_npy/mentor_conv1_1_iteration"+str(iteration)+".npy", mentor_conv1_1)
        np.save("output_vgg16/filters_npy/mentor_conv1_2_iteration"+str(iteration)+".npy", mentor_conv1_2)
        np.save("output_vgg16/filters_npy/mentor_conv2_1_iteration"+str(iteration)+".npy", mentor_conv2_1)
        np.save("output_vgg16/filters_npy/mentor_conv2_2_iteration"+str(iteration)+".npy", mentor_conv2_2)
        np.save("output_vgg16/filters_npy/mentor_conv3_1_iteration"+str(iteration)+".npy", mentor_conv3_1)
        np.save("output_vgg16/filters_npy/mentor_conv3_2_iteration"+str(iteration)+".npy", mentor_conv3_2)
        np.save("output_vgg16/filters_npy/mentor_conv3_3_iteration"+str(iteration)+".npy", mentor_conv3_3)
        np.save("output_vgg16/filters_npy/mentor_conv4_1_iteration"+str(iteration)+".npy", mentor_conv4_1)
        np.save("output_vgg16/filters_npy/mentor_conv4_2_iteration"+str(iteration)+".npy", mentor_conv4_2)
        np.save("output_vgg16/filters_npy/mentor_conv4_3_iteration"+str(iteration)+".npy", mentor_conv4_3)
        np.save("output_vgg16/filters_npy/mentor_conv5_1_iteration"+str(iteration)+".npy", mentor_conv5_1)
        np.save("output_vgg16/filters_npy/mentor_conv5_2_iteration"+str(iteration)+".npy", mentor_conv5_2)
        np.save("output_vgg16/filters_npy/mentor_conv5_3_iteration"+str(iteration)+".npy", mentor_conv5_3)
        np.save("output_vgg16/filters_npy/mentor_fc1_iteration"+str(iteration)+".npy", mentor_fc1)
        np.save("output_vgg16/filters_npy/mentor_fc2_iteration"+str(iteration)+".npy", mentor_fc2)
        np.save("output_vgg16/filters_npy/mentor_fc3_iteration"+str(iteration)+".npy", mentor_fc3)

    def train_model(self, data_input_train, data_input_test, images_placeholder, labels_placeholder, sess):

        try:
            print('Begin to train model.....................................................')

            eval_correct = self.evaluation(self.softmax, labels_placeholder)

            for i in range(NUM_ITERATIONS):

                feed_dict, images_feed, labels_feed = self.fill_feed_dict(data_input_train, images_placeholder, labels_placeholder, sess)

                if FLAGS.student or FLAGS.teacher:

                    _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if i % 100 == 0:
                        print ('Step %d: loss_value = %.20f' % (i, loss_value))

                if FLAGS.dependent_student:

                    # self.compute_0filter_of_teacherOutput(sess, feed_dict, images_feed, i)
                    # self.cosine = cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict, sess, feed_dict, FLAGS.num_optimizers)
                    # cosine = sess.run(self.cosine, feed_dict=feed_dict)
                    # self.select_optimizers_and_loss(cosine)
                    """
                    if FLAGS.fitnet_phase2:
                        lamma_decay_rate = (FLAGS.lamma_KD_initial - 1.0) / NUM_ITERATIONS
                        lamma = FLAGS.lamma_KD_initial - lamma_decay_rate * i
                        self.lamma_KD.load(lamma, session=sess)
                        if i % 300 == 0:
                            print('lamma_KD of {} for iteration {}'.format(lamma, i))
                    """

                    _, self.loss_value_list = sess.run([self.train_op_list, self.loss_list], feed_dict=feed_dict)

                    if i % 100 == 0:
                        if FLAGS.proposed_method:
                            print('Step %d: loss_value1 = %.20f' % (i, self.loss_value_list[0]))
                            print('Step %d: loss_value2 = %.20f' % (i, self.loss_value_list[1]))
                            #print('Step %d: loss_value3 = %.20f' % (i, self.loss_value_list[2]))
                            #print('Step %d: loss_value4 = %.20f' % (i, self.loss_value_list[3]))
                            #print('Step %d: loss_value5 = %.20f' % (i, self.loss_value_list[4]))
                            #print('Step %d: loss_softmax = %.20f' % (i, self.loss_value_list[0]))
                            print('Step %d: loss_fc = %.20f' % (i, self.loss_value_list[2]))
                        elif FLAGS.fitnet_phase1:
                            print('Step %d: loss_value_fitnet_phase1 = %.20f' % (i, self.loss_value_list[0]))
                        elif FLAGS.fitnet_phase2:
                            print('Step %d: loss_value_fitnet_phase2 = %.20f' % (i, self.loss_value_list[0]))
                        else:
                            raise ValueError("Not found method")
                        print ("\n")


                if (i) % (FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size) == 0 or (i) == NUM_ITERATIONS - 1:

                    # checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')

                    #if FLAGS.teacher:
                    #    print("save teacher to: "+str(FLAGS.teacher_weights_filename))
                    #    self.saver.save(sess, FLAGS.teacher_weights_filename)
                    #if FLAGS.student:
                    #    print("Save student weights to: "+str(FLAGS.student_filename))
                    #    self.saver.save(sess, FLAGS.student_filename)
                    #if FLAGS.fitnet_phase1:
                    #    saver_new = tf.train.Saver()
                    #    saver_new.save(sess, FLAGS.fitnet_phase1_filename)
                    #    print("save fitnet_phase1 to: "+str(FLAGS.fitnet_phase1_filename))

                    # check learning rate
                    num_epoch = int(i / float(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size))
                    learning_rate = sess.run(self.lr)
                    print('Epoch is: {}, learning rate is: {}'.format(num_epoch, learning_rate))

                    print ("Training Data Eval:")
                    self.do_eval(sess,eval_correct,self.softmax,images_placeholder,labels_placeholder,data_input_train,'Train')

                    print ("Test  Data Eval:")
                    self.do_eval(sess,eval_correct,self.softmax,images_placeholder,labels_placeholder,data_input_test,'Test')
                    print ("max test accuracy % f", max(test_accuracy_list))

                    print(test_accuracy_list)

        except Exception as e:
            print(e)

    def main(self, _):
        start_time = time.time()
        with tf.Graph().as_default():

            print("test whether to use gpu")
            print(device_lib.list_local_devices())

            # This line allows the code to use only sufficient memory and does not block entire GPU
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

            # set the seed so that we have same loss values and initializations for every run.
            tf.set_random_seed(seed)

            data_input_train = DataInput(dataset_path, FLAGS.train_dataset, FLAGS.batch_size,
                                         FLAGS.num_training_examples, FLAGS.image_width, FLAGS.image_height,
                                         FLAGS.num_channels, seed, FLAGS.dataset)

            data_input_test = DataInput(dataset_path, FLAGS.test_dataset, FLAGS.batch_size, FLAGS.num_testing_examples,
                                        FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

            #data_input_validation = DataInput(dataset_path, FLAGS.validation_dataset, FLAGS.batch_size,
            #                                  FLAGS.num_validation_examples, FLAGS.image_width, FLAGS.image_height,
            #                                  FLAGS.num_channels, seed, FLAGS.dataset)

            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels))
            labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))

            # config = tf.ConfigProto(allow_soft_placement=True)
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            # config.gpu_options.per_process_gpu_memory_fraction = 0.90

            sess = tf.Session(config=config)
            ## this line is used to enable tensorboard debugger
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
            #summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            #phase_train = tf.placeholder(tf.bool, name='phase_train')
            #summary = tf.summary.merge_all()

            print("NUM_ITERATIONS: "+str(NUM_ITERATIONS))
            print("learning_rate: " + str(FLAGS.learning_rate))
            print("batch_size: " + str(FLAGS.batch_size))

            if FLAGS.student:
                self.define_independent_student(images_placeholder, labels_placeholder, seed, global_step, sess)

            elif FLAGS.teacher:
                self.define_teacher(images_placeholder, labels_placeholder, global_step, sess)

            elif FLAGS.dependent_student:
                self.define_dependent_student(images_placeholder, labels_placeholder, seed, global_step,sess)

            self.train_model(data_input_train, data_input_test, images_placeholder, labels_placeholder, sess)

            #writer_tensorboard = tf.summary.FileWriter('tensorboard/', sess.graph)

            coord.request_stop()
            coord.join(threads)

        sess.close()
        #writer_tensorboard.close()

        end_time = time.time()
        runtime = round((end_time - start_time) / (60 * 60), 2)
        print("run time is: " + str(runtime) + " hour")
        print("1th: "+ str(count_cosine[0]) + "," + str(count_cosine[1]))
        print("2th: "+ str(count_cosine[2]) + "," + str(count_cosine[3]))
        print("3th: "+ str(count_cosine[4]) + "," + str(count_cosine[5])+ "," + str(count_cosine[6]))
        print("4th: "+ str(count_cosine[7]) + "," + str(count_cosine[8])+ "," + str(count_cosine[9]))
        print("5th: "+ str(count_cosine[10]) + "," + str(count_cosine[11])+ "," + str(count_cosine[12]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher',type=bool,help='train teacher',default=False)
    parser.add_argument('--dependent_student',type=bool,help='train dependent student',default=False)
    parser.add_argument('--student',type=bool,help='train independent student',default=False)
    parser.add_argument('--fitnet_phase1',type=bool,help='fitnet_phase1',default=False)
    parser.add_argument('--fitnet_phase2',type=bool,help='fitnet_phase1',default=False)
    parser.add_argument('--proposed_method',type=bool,help='proposed_method',default=False)
    parser.add_argument('--teacher_weights_filename',type=str,default="./summary-log/teacher_weights_filename_caltech101")
    parser.add_argument('--student_filename',type=str,default="./summary-log/####")
    parser.add_argument('--learning_rate',type=float,default=0.0001)
    parser.add_argument('--batch_size',type=int,default=25)
    parser.add_argument('--image_height',type=int,default=224)
    parser.add_argument('--image_width',type=int,default=224)
    parser.add_argument('--train_dataset',type=str,default="dataset_input/caltech101-train.txt")
    parser.add_argument('--test_dataset',type=str,default="dataset_input/caltech101-test.txt")
    parser.add_argument('--validation_dataset',type=str,default="dataset_input/caltech101-validation.txt")
    parser.add_argument('--temp_softmax',type=int,default=1)
    parser.add_argument('--num_classes',type=int,default=102)
    parser.add_argument('--learning_rate_pretrained',type=float,default=0.0001)
    parser.add_argument('--NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN',type=int,default=5853)
    parser.add_argument('--num_training_examples',type=int,default=5853)
    parser.add_argument('--num_testing_examples',type=int,default=1829)
    parser.add_argument('--num_validation_examples',type=int,default=1463)
    parser.add_argument('--dataset',type=str,help='name of the dataset',default='caltech101')
    parser.add_argument('--num_channels',type=int,help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',default='3')
    parser.add_argument('--top_1_accuracy',type=bool,help='top-1-accuracy',default=False)
    parser.add_argument('--top_3_accuracy',type=bool,help='top-3-accuracy',default=False)
    parser.add_argument('--top_5_accuracy',type=bool,help='top-5-accuracy',default=False)
    parser.add_argument('--num_iterations',type=int,help='num_iterations',default=1)
    parser.add_argument('--interval_output_train',type=bool,help='interval_output_train',default=False)
    parser.add_argument('--interval_lossValue_train',type=bool,help='interval_lossValue_train',default=False)
    parser.add_argument('--initialization',type=bool,help='initialization',default=False)
    parser.add_argument('--num_optimizers',type=int,help='number of mapping layers from teacher',default=5)
    parser.add_argument('--fitnet_phase1_filename',type=str,help='save dependent student of fitnet_phase1',default="./summary-log/fitnet/")
    parser.add_argument('--lamma_KD_initial',type=float,default=4.0)

    FLAGS, unparsed = parser.parse_known_args()
    ex = VGG16()
    tf.app.run(main=ex.main, argv=[sys.argv[0]] + unparsed)
