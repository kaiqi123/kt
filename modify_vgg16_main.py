import tensorflow as tf
import numpy as np
import random
from DataInput import DataInput
from vgg16mentee_temp import Mentee
#from vgg16mentee import Mentee
from vgg16mentor import Mentor
from vgg16embed import Embed
from mentor import Teacher
import os
import time
import pdb
import sys
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from PIL import Image
import argparse
import csv
from tensorflow.python.client import device_lib
from compute_cosine_similarity import cosine_similarity_of_same_width

dataset_path = "./"
tf.reset_default_graph()
NUM_ITERATIONS = 7820
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
    def fill_feed_dict(self, data_input, images_pl, labels_pl, sess, mode, phase_train):
        """
        Based on the mode whether it is train, test or validation; we fill the feed_dict with appropriate images and labels.
        Args:
            data_input: object instantiated for DataInput class
            images_pl: placeholder to hold images of the datasets
            labels_pl: placeholder to hold labels of the datasets
            mode: mode is either train or test or validation


        Returns:
            feed_dict: dictionary consists of images placeholder, labels placeholder and phase_train as keys
                       and images, labels and a boolean value phase_train as values.

        """

        images_feed, labels_feed = sess.run([data_input.example_batch, data_input.label_batch])

        #print(images_feed.shape)
        #print(len(labels_feed))

        if mode == 'Train':
            feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
                phase_train: True
            }

        if mode == 'Test':
            feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
                phase_train: False
            }

        if mode == 'Validation':
            feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
                phase_train: False
            }
        return feed_dict, images_feed, labels_feed

    def evaluation(self, logits, labels):

        if FLAGS.top_1_accuracy:
            print('evaluation: top 1 accuracy ')
            correct = tf.nn.in_top_k(logits, labels, 1)
        elif FLAGS.top_3_accuracy:
            correct = tf.nn.in_top_k(logits, labels, 3)
        elif FLAGS.top_5_accuracy:
            correct = tf.nn.in_top_k(logits, labels, 5)

        return tf.reduce_sum(tf.cast(correct, tf.int32))


    def evaluation_teacher(self, logits, labels):

        if FLAGS.top_1_accuracy:
            print('evaluation: top 1 accuracy ')
            correct = tf.nn.in_top_k(logits, labels, 1)
        elif FLAGS.top_3_accuracy:
            correct = tf.nn.in_top_k(logits, labels, 3)
        elif FLAGS.top_5_accuracy:
            correct = tf.nn.in_top_k(logits, labels, 5)

        return tf.cast(correct, tf.int32)

    def do_eval(self, sess, eval_correct, logits, images_placeholder, labels_placeholder, dataset,mode, phase_train):

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
                    feed_dict, images_feed, labels_feed = self.fill_feed_dict(dataset, images_placeholder,
                                                            labels_placeholder,sess, mode,phase_train)
                count = sess.run(eval_correct, feed_dict=feed_dict)
                true_count = true_count + count

            precision = float(true_count) / num_examples
            print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
                            (num_examples, true_count, precision))

            if mode == 'Test':
                test_accuracy_list.append(precision)

    def get_mentor_variables_to_restore(self):

        """
        Returns:: names of the weights and biases of the teacher model

        """
        return [var for var in tf.global_variables() if var.op.name.startswith("mentor") and (var.op.name.endswith("biases") or var.op.name.endswith("weights")) and (var.op.name != ("mentor_fc3/mentor_weights") and  var.op.name != ("mentor_fc3/mentor_biases"))]

    def caculate_rmse_loss(self):

        """
        Here layers of same width are mapped together.
        """

        #self.softloss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.softmax, self.mentee_data_dict.softmax))))
        #self.loss_fc3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.fc3l, self.mentee_data_dict.fc3l))))

        y_pred_tf = tf.convert_to_tensor(self.mentor_data_dict.softmax, np.float32)
        y_true_tf = tf.convert_to_tensor(self.mentee_data_dict.softmax, np.float32)
        eps = 1e-6
        cliped_y_pref_tf = tf.clip_by_value(y_pred_tf, eps, 1 - eps)
        self.loss_softCrossEntropy = tf.reduce_mean(-tf.reduce_sum(y_true_tf * tf.log(cliped_y_pref_tf), axis=1))

        self.l1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv1_2, self.mentee_data_dict.conv1_1))))
        if FLAGS.num_optimizers >= 2:
            self.l2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv2_1, self.mentee_data_dict.conv2_1))))
        if FLAGS.num_optimizers >= 3:
            self.l3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv3_1, self.mentee_data_dict.conv3_1))))
        if FLAGS.num_optimizers >= 4:
            self.l4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv4_2, self.mentee_data_dict.conv4_1))))
        if FLAGS.num_optimizers == 5:
            self.l5 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv5_2, self.mentee_data_dict.conv5_1))))

        """
        self.l11 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv1_1, self.mentee_data_dict.conv1_1))))
        self.l12 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv1_2, self.mentee_data_dict.conv1_1))))

        self.l21 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv2_1, self.mentee_data_dict.conv2_1))))
        self.l22 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv2_2, self.mentee_data_dict.conv2_1))))

        self.l31 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv3_1, self.mentee_data_dict.conv3_1))))
        self.l32 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv3_2, self.mentee_data_dict.conv3_1))))
        self.l33 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv3_3, self.mentee_data_dict.conv3_1))))

        if FLAGS.num_optimizers == 5:
            self.l41 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv4_1, self.mentee_data_dict.conv4_1))))
            self.l42 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv4_2, self.mentee_data_dict.conv4_1))))
            self.l43 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv4_3, self.mentee_data_dict.conv4_1))))

            self.l51 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv5_1, self.mentee_data_dict.conv5_1))))
            self.l52 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv5_2, self.mentee_data_dict.conv5_1))))
            self.l53 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv5_3, self.mentee_data_dict.conv5_1))))
        """
    def define_multiple_optimizers(self, lr):

        print("define multiple optimizers")

        #self.train_op_soft = tf.train.AdamOptimizer(lr).minimize(self.softloss)
        #self.train_op_fc3 = tf.train.AdamOptimizer(lr).minimize(self.loss_fc3)
        self.train_op_softCrossEntropy = tf.train.AdamOptimizer(lr).minimize(self.loss_softCrossEntropy)

        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)

        l1_var_list = []
        l1_var_list.append([var for var in tf.global_variables() if var.op.name == "mentee_conv1_1/mentee_weights"][0])
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(self.l1, var_list=l1_var_list)

        if FLAGS.num_optimizers >= 2:
            l2_var_list = []
            l2_var_list.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])
            self.train_op2 = tf.train.AdamOptimizer(lr).minimize(self.l2, var_list=l2_var_list)

        if FLAGS.num_optimizers >= 3:
            l3_var_list = []
            l3_var_list.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
            self.train_op3 = tf.train.AdamOptimizer(lr).minimize(self.l3, var_list=l3_var_list)

        if FLAGS.num_optimizers >= 4:
            l4_var_list = []
            l4_var_list.append([var for var in tf.global_variables() if var.op.name=="mentee_conv4_1/mentee_weights"][0])
            self.train_op4 = tf.train.AdamOptimizer(lr).minimize(self.l4, var_list=l4_var_list)

        if FLAGS.num_optimizers == 5:
            l5_var_list = []
            l5_var_list.append([var for var in tf.global_variables() if var.op.name=="mentee_conv5_1/mentee_weights"][0])
            self.train_op5 = tf.train.AdamOptimizer(lr).minimize(self.l5, var_list=l5_var_list)

        """
        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.train_op11 = tf.train.AdamOptimizer(lr).minimize(self.l11, var_list=l1_var_list)
        self.train_op12 = tf.train.AdamOptimizer(lr).minimize(self.l12, var_list=l1_var_list)

        self.train_op21 = tf.train.AdamOptimizer(lr).minimize(self.l21, var_list=l2_var_list)
        self.train_op22 = tf.train.AdamOptimizer(lr).minimize(self.l22, var_list=l2_var_list)

        self.train_op31 = tf.train.AdamOptimizer(lr).minimize(self.l31, var_list=l3_var_list)
        self.train_op32 = tf.train.AdamOptimizer(lr).minimize(self.l32, var_list=l3_var_list)
        self.train_op33 = tf.train.AdamOptimizer(lr).minimize(self.l33, var_list=l3_var_list)

        if FLAGS.num_optimizers == 5:
            self.train_op41 = tf.train.AdamOptimizer(lr).minimize(self.l41, var_list=l4_var_list)
            self.train_op42 = tf.train.AdamOptimizer(lr).minimize(self.l42, var_list=l4_var_list)
            self.train_op43 = tf.train.AdamOptimizer(lr).minimize(self.l43, var_list=l4_var_list)

            self.train_op51 = tf.train.AdamOptimizer(lr).minimize(self.l51, var_list=l5_var_list)
            self.train_op52 = tf.train.AdamOptimizer(lr).minimize(self.l52, var_list=l5_var_list)
            self.train_op53 = tf.train.AdamOptimizer(lr).minimize(self.l53, var_list=l5_var_list)
        """

    def define_independent_student(self, images_placeholder, labels_placeholder, seed, phase_train, global_step, sess):

        """
            Student is trained without taking knowledge from teacher

            Args:
                images_placeholder: placeholder to hold images of dataset
                labels_placeholder: placeholder to hold labels of the images of the dataset
                seed: seed value to have sequence in the randomness
                phase_train: determines test or train state of the network
        """

        student = Mentee(FLAGS.num_channels)
        print("Independent student")
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        ## number of steps after which learning rate should decay
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        if FLAGS.num_optimizers == 5:
            mentee_data_dict = student.build_conv5fc2(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        if FLAGS.num_optimizers == 4:
            mentee_data_dict = student.build_conv4fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        if FLAGS.num_optimizers == 3:
            mentee_data_dict = student.build_conv3fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        if FLAGS.num_optimizers == 2:
            mentee_data_dict = student.build_conv2fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        if FLAGS.num_optimizers == 1:
            mentee_data_dict = student.build_conv1fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)

        self.loss = student.loss(labels_placeholder)
        ## learning rate is decayed exponentially with a decay factor of 0.9809 after every epoch
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        self.train_op = student.training(self.loss, lr, global_step)
        self.softmax = mentee_data_dict.softmax
        # initialize all the variables of the network
        init = tf.initialize_all_variables()
        sess.run(init)
        ## saver object is created to save all the variables to a file
        self.saver = tf.train.Saver()

    def define_teacher(self, images_placeholder, labels_placeholder, phase_train, global_step, sess):

        """
            1. Train teacher prior to student so that knowledge from teacher can be transferred to train student.
            2. Teacher object is trained by importing weights from a pretrained vgg 16 network
            3. Mentor object is a network trained from scratch. We did not find the pretrained network with the same architecture for cifar10.
               Thus, trained the network from scratch on cifar10

        """

        if FLAGS.dataset == 'cifar10' or 'mnist':
            print("Train Teacher (cifar10 or mnist)")
            mentor = Teacher()
        if FLAGS.dataset == 'caltech101':
            print("Train Teacher (caltech101)")
            mentor = Mentor()

        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        mentor_data_dict = mentor.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, phase_train)
        self.loss = mentor.loss(labels_placeholder)
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        if FLAGS.dataset == 'caltech101':
            ## restore all the weights
            variables_to_restore = self.get_mentor_variables_to_restore()
            self.train_op = mentor.training(self.loss, FLAGS.learning_rate_pretrained, lr, global_step,
                                            variables_to_restore, mentor.get_training_vars())
        if FLAGS.dataset == 'cifar10':
            print("cifar10")
            self.train_op = mentor.training(self.loss, FLAGS.learning_rate, global_step)

        self.softmax = mentor_data_dict.softmax
        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver = tf.train.Saver()


    def define_dependent_student(self, images_placeholder, labels_placeholder, phase_train, seed, global_step, sess):

        """
        Student is trained by taking supervision from teacher for every batch of data
        Same batch of input data is passed to both teacher and student for every iteration
        """

        if FLAGS.dataset == 'cifar10':
            print("Train dependent student (cifar10 or mnist)")
            vgg16_mentor = Teacher(False)
        if FLAGS.dataset == 'caltech101':
            print("Train dependent student (caltech101)")
            vgg16_mentor = Mentor(False)

        vgg16_mentee = Mentee(FLAGS.num_channels)
        self.mentor_data_dict = vgg16_mentor.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax,
                                                   phase_train)

        if FLAGS.num_optimizers == 5:
            self.mentee_data_dict = vgg16_mentee.build_conv5fc2(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        if FLAGS.num_optimizers == 4:
            self.mentee_data_dict = vgg16_mentee.build_conv4fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        if FLAGS.num_optimizers == 3:
            self.mentee_data_dict = vgg16_mentee.build_conv3fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        if FLAGS.num_optimizers == 2:
            self.mentee_data_dict = vgg16_mentee.build_conv2fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        if FLAGS.num_optimizers == 1:
            self.mentee_data_dict = vgg16_mentee.build_conv1fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)


        self.softmax = self.mentee_data_dict.softmax
        mentor_variables_to_restore = self.get_mentor_variables_to_restore()
        self.loss = vgg16_mentee.loss(labels_placeholder)
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        self.caculate_rmse_loss()
        self.define_multiple_optimizers(lr)

        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver(mentor_variables_to_restore)
        saver.restore(sess, FLAGS.teacher_weights_filename)
        #saver.restore(sess, "./summary-log/new_method_teacher_weights_filename_caltech101_clean_code")

        if FLAGS.initialization:
            for var in tf.global_variables():
                if var.op.name == "mentor_conv1_1/mentor_weights":
                    print("initialization: conv1_1")
                    self.mentee_data_dict.parameters[0].assign(var.eval(session=sess)).eval(session=sess)

                if FLAGS.num_optimizers >= 2:
                    if var.op.name == "mentor_conv2_1/mentor_weights":
                        print("initialization: conv2_1")
                        self.mentee_data_dict.parameters[2].assign(var.eval(session=sess)).eval(session=sess)

                if FLAGS.num_optimizers >= 3:
                    if var.op.name == "mentor_conv3_1/mentor_weights":
                        print("initialization: conv3_1")
                        self.mentee_data_dict.parameters[4].assign(var.eval(session=sess)).eval(session=sess)

                if FLAGS.num_optimizers >= 4:
                    if var.op.name == "mentor_conv4_1/mentor_weights":
                        print("initialization: conv4_1")
                        self.mentee_data_dict.parameters[6].assign(var.eval(session=sess)).eval(session=sess)

                if FLAGS.num_optimizers == 5:
                    if var.op.name == "mentor_conv5_1/mentor_weights":
                        print("initialization: conv5_1")
                        self.mentee_data_dict.parameters[8].assign(var.eval(session=sess)).eval(session=sess)

                    if var.op.name == "mentor_fc1/mentor_weights":
                        self.mentee_data_dict.parameters[10].assign(var.eval(session=sess)).eval(session=sess)

                    if var.op.name == "mentor_fc3/mentor_weights":
                        self.mentee_data_dict.parameters[12].assign(var.eval(session=sess)).eval(session=sess)


    def select_optimizers_and_loss(self,cosine):
        #print(cosine)
        if cosine[0] == 1:
            #print("1:11")
            self.train_op1 = self.train_op11
            self.l1 = self.l11
            count_cosine[0]=count_cosine[0]+1
        else:
            #print("1:222")
            self.train_op1 = self.train_op12
            self.l1 = self.l12
            count_cosine[1] = count_cosine[1] + 1

        if cosine[1] == 1:
            #print("2:11")
            self.train_op2 = self.train_op21
            self.l2 = self.l21
            count_cosine[2] = count_cosine[2] + 1
        else:
            #print("2:222")
            self.train_op2 = self.train_op22
            self.l2 = self.l22
            count_cosine[3] = count_cosine[3] + 1

        if cosine[2] == 1:
            #print("3:11")
            self.train_op3 = self.train_op31
            self.l3 = self.l31
            count_cosine[4] = count_cosine[4] + 1
        elif cosine[2] == 2:
            #print("3:222")
            self.train_op3 = self.train_op32
            self.l3 = self.l32
            count_cosine[5] = count_cosine[5] + 1
        else:
            #print("3:333")
            self.train_op3 = self.train_op33
            self.l3 = self.l33
            count_cosine[6] = count_cosine[6] + 1

        if FLAGS.num_optimizers == 5:

            if cosine[3] == 1:
                #print("4:11")
                self.train_op4 = self.train_op41
                self.l4 = self.l41
                count_cosine[7] = count_cosine[7] + 1
            elif cosine[3] == 2:
                #print("4:222")
                self.train_op4 = self.train_op42
                self.l4 = self.l42
                count_cosine[8] = count_cosine[8] + 1
            else:
                #print("4:33")
                self.train_op4 = self.train_op43
                self.l4 = self.l43
                count_cosine[9] = count_cosine[9] + 1

            if cosine[4] == 1:
                #print("5:11")
                self.train_op5 = self.train_op51
                self.l5 = self.l51
                count_cosine[10] = count_cosine[10] + 1
            elif cosine[4] == 2:
                #print("5:222")
                self.train_op5 = self.train_op52
                self.l5 = self.l52
                count_cosine[11] = count_cosine[11] + 1
            else:
                #print("5:333")
                self.train_op5 = self.train_op53
                self.l5 = self.l53
                count_cosine[12] = count_cosine[12] + 1


    def run_dependent_student(self, feed_dict, sess, i):



        #print("connect teacher: "+str(i))

        #self.cosine = cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict, sess, feed_dict, FLAGS.num_optimizers)
        #cosine = sess.run(self.cosine, feed_dict=feed_dict)
        #self.select_optimizers_and_loss(cosine)

        #_, self.loss_value_soft = sess.run([self.train_op_soft, self.softloss], feed_dict=feed_dict)
        #_, self.loss_value_fc3 = sess.run([self.train_op_fc3, self.loss_fc3], feed_dict=feed_dict)
        #_, self.loss_value_softCrossEntropy = sess.run([self.train_op_softCrossEntropy, self.loss_softCrossEntropy], feed_dict=feed_dict)

        _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
        _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
        if FLAGS.num_optimizers >= 2:
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
        if FLAGS.num_optimizers >= 3:
            _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
        if FLAGS.num_optimizers >= 4:
            _, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)
        if FLAGS.num_optimizers == 5:
            _, self.loss_value5 = sess.run([self.train_op5, self.l5], feed_dict=feed_dict)


        """
        subtract = tf.subtract(self.mentor_data_dict.softmax, self.mentee_data_dict.softmax)
        square = tf.square(subtract)
        mean = tf.reduce_mean(square)
        loss = tf.sqrt(mean)
        subtract, square, mean, loss = sess.run([subtract, square, mean, loss], feed_dict=feed_dict)

        print(subtract.shape)
        print(square.shape)
        print(mean)
        print(loss)
        """



    def train_model(self, data_input_train, data_input_test, images_placeholder, labels_placeholder, sess,
                    phase_train):

        try:
            print('train model')

            eval_correct = self.evaluation(self.softmax, labels_placeholder)

            if FLAGS.dependent_student:
                teacher_eval_correct = self.evaluation_teacher(self.mentor_data_dict.softmax, labels_placeholder)
                teacher_truecount_perEpoch_list = []

            for i in range(NUM_ITERATIONS):

                #print("iteration: "+str(i))

                feed_dict, images_feed, labels_feed = self.fill_feed_dict(data_input_train, images_placeholder,
                                                labels_placeholder, sess, 'Train', phase_train)

                if FLAGS.student or FLAGS.teacher:

                    #print("train function: independent student or teacher")
                    _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                    if i % 10 == 0:
                        print ('Step %d: loss_value = %.20f' % (i, loss_value))

                if FLAGS.dependent_student:

                    if (i % FLAGS.num_iterations == 0):

                        print("connect to teacher: "+ str(i))

                        teacher_eval_correct_array= sess.run(teacher_eval_correct, feed_dict=feed_dict)
                        teacher_eval_correct_list = list(teacher_eval_correct_array)
                        #print(teacher_eval_correct_list)
                        #print(labels_feed)

                        count0 = teacher_eval_correct_list.count(0)
                        index1 = teacher_eval_correct_list.index(1)
                        if count0>0:
                            #print(count0)
                            labels_feed_new = []
                            images_feed_new = []
                            k = 0
                            for j in range(FLAGS.batch_size):
                                if teacher_eval_correct_array[j] == 1:
                                    labels_feed_new.append(labels_feed[j])
                                    images_feed_new.append(images_feed[j])
                                else:
                                    if len(labels_feed_new)==0:
                                        labels_feed_new.append(labels_feed[index1])
                                        images_feed_new.append(images_feed[index1])
                                    else:
                                        labels_feed_new.append(labels_feed_new[k])
                                        images_feed_new.append(images_feed_new[k])
                                        k = k + 1

                            labels_feed_new = np.array(labels_feed_new)
                            images_feed_new = np.array(images_feed_new)
                            #print(labels_feed_new)

                            feed_dict_new = {
                                images_placeholder: images_feed_new,
                                labels_placeholder: labels_feed_new,
                                phase_train: True
                            }
                            self.run_dependent_student(feed_dict_new, sess, i)
                        else:
                            self.run_dependent_student(feed_dict, sess, i)

                        teacher_truecount_perEpoch = sum(teacher_eval_correct_list)
                        teacher_truecount_perEpoch_list.append(teacher_truecount_perEpoch)

                    else:
                        _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)


                    #print("iteration222: " + str(i))
                    if i % 10 == 0:
                        # print("train function: dependent student, multiple optimizers")
                        #print ('Step %d: loss_value_soft = %.20f' % (i, self.loss_value_soft))
                        #print ('Step %d: loss_value_fc3 = %.20f' % (i, self.loss_value_fc3))
                        #print ('Step %d: loss_value_softCrossEntropy = %.20f' % (i, self.loss_value_softCrossEntropy))

                        print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                        print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                        if FLAGS.num_optimizers >= 2:
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                        if FLAGS.num_optimizers >= 3:
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                        if FLAGS.num_optimizers >= 4:
                            print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                        if FLAGS.num_optimizers == 5:
                            print ('Step %d: loss_value5 = %.20f' % (i, self.loss_value5))
                        print ("\n")
                    

                if (i) % (FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size) == 0 or (i) == NUM_ITERATIONS - 1:
                #if (i) % 10 == 0 or (i) == NUM_ITERATIONS - 1:
                    # checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')

                    if FLAGS.teacher:
                        print("save teacher to: "+str(FLAGS.teacher_weights_filename))
                        self.saver.save(sess, FLAGS.teacher_weights_filename)
                    #elif FLAGS.student:
                    #    saver.save(sess, FLAGS.student_filename)
                    elif FLAGS.dependent_student:
                        saver_new = tf.train.Saver()
                        saver_new.save(sess, FLAGS.dependent_student_filename)

                    if FLAGS.dependent_student:
                        print(teacher_truecount_perEpoch_list)
                        teacher_alltrue = teacher_truecount_perEpoch_list.count(FLAGS.batch_size)
                        teacher_alltrue_list.append(teacher_alltrue)

                        teacher_alltrue_list_127.append(teacher_truecount_perEpoch_list.count(FLAGS.batch_size-1))
                        teacher_alltrue_list_126.append(teacher_truecount_perEpoch_list.count(FLAGS.batch_size-2))

                        print("teacher_alltrue_list" + str(FLAGS.batch_size)+":"+str(teacher_alltrue_list))
                        print("teacher_alltrue_list" + str(FLAGS.batch_size-1)+":"+str(teacher_alltrue_list_127))
                        print("teacher_alltrue_list" + str(FLAGS.batch_size-2)+":" + str(teacher_alltrue_list_126))

                        teacher_truecount_perEpoch_list = []

                    print ("Training Data Eval:")
                    self.do_eval(sess,
                                 eval_correct,
                                 self.softmax,
                                 images_placeholder,
                                 labels_placeholder,
                                 data_input_train,
                                 'Train', phase_train)

                    print ("Test  Data Eval:")
                    self.do_eval(sess,
                                 eval_correct,
                                 self.softmax,
                                 images_placeholder,
                                 labels_placeholder,
                                 data_input_test,
                                 'Test', phase_train)
                    print ("max test accuracy % f", max(test_accuracy_list))




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

            images_placeholder = tf.placeholder(tf.float32,
                                                shape=(FLAGS.batch_size, FLAGS.image_height,
                                                       FLAGS.image_width, FLAGS.num_channels))
            labels_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size))

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
            phase_train = tf.placeholder(tf.bool, name='phase_train')
            #summary = tf.summary.merge_all()

            print("NUM_ITERATIONS: "+str(NUM_ITERATIONS))
            print("learning_rate: " + str(FLAGS.learning_rate))
            print("batch_size: " + str(FLAGS.batch_size))
            print("teacher_weights_filename: "+FLAGS.teacher_weights_filename)

            if FLAGS.student:
                self.define_independent_student(images_placeholder, labels_placeholder, seed, phase_train, global_step,
                                               sess)

            elif FLAGS.teacher:
                self.define_teacher(images_placeholder, labels_placeholder, phase_train, global_step, sess)

            elif FLAGS.dependent_student:
                self.define_dependent_student(images_placeholder, labels_placeholder, phase_train, seed, global_step,
                                             sess)

            self.train_model(data_input_train, data_input_test, images_placeholder, labels_placeholder, sess,
                             phase_train)

            print(test_accuracy_list)
            writer_tensorboard = tf.summary.FileWriter('tensorboard/', sess.graph)

            coord.request_stop()
            coord.join(threads)

        sess.close()
        writer_tensorboard.close()

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
    parser.add_argument(
        '--teacher',
        type=bool,
        help='train teacher',
        default=False
    )
    parser.add_argument(
        '--dependent_student',
        type=bool,
        help='train dependent student',
        default=False
    )
    parser.add_argument(
        '--student',
        type=bool,
        help='train independent student',
        default=False
    )
    parser.add_argument(
        '--teacher_weights_filename',
        type=str,
        default="./summary-log/new_method_teacher_weights_filename_caltech101"
    )
    parser.add_argument(
        '--student_filename',
        type=str,
        default="./summary-log/new_method_student_weights_filename_caltech101"
    )
    parser.add_argument(
        '--dependent_student_filename',
        type=str,
        default="./summary-log/new_method_dependent_student_weights_filename_caltech101"
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=25
    )
    parser.add_argument(
        '--image_height',
        type=int,
        default=224
    )
    parser.add_argument(
        '--image_width',
        type=int,
        default=224
    )
    parser.add_argument(
        '--train_dataset',
        type=str,
        default="dataset_input/caltech101-train.txt"
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        default="dataset_input/caltech101-test.txt"
    )
    parser.add_argument(
        '--validation_dataset',
        type=str,
        default="dataset_input/caltech101-validation.txt"
    )
    parser.add_argument(
        '--temp_softmax',
        type=int,
        default=1
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=102
    )
    parser.add_argument(
        '--learning_rate_pretrained',
        type=float,
        default=0.0001
    )
    parser.add_argument(
        '--NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN',
        type=int,
        default=5853
    )
    parser.add_argument(
        '--num_training_examples',
        type=int,
        default=5853
    )
    parser.add_argument(
        '--num_testing_examples',
        type=int,
        default=1829
    )
    parser.add_argument(
        '--num_validation_examples',
        type=int,
        default=1463
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='name of the dataset',
        default='caltech101'
    )
    parser.add_argument(
        '--mnist_data_dir',
        type=str,
        help='name of the dataset',
        default='./mnist_data'
    )
    parser.add_argument(
        '--num_channels',
        type=int,
        help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
        default='3'
    )
    parser.add_argument(
        '--top_1_accuracy',
        type=bool,
        help='top-1-accuracy',
        default=False
    )
    parser.add_argument(
        '--top_3_accuracy',
        type=bool,
        help='top-3-accuracy',
        default=False
    )
    parser.add_argument(
        '--top_5_accuracy',
        type=bool,
        help='top-5-accuracy',
        default=False
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        help='num_iterations',
        default=1
    )
    parser.add_argument(
        '--interval_output_train',
        type=bool,
        help='interval_output_train',
        default=False
    )
    parser.add_argument(
        '--interval_lossValue_train',
        type=bool,
        help='interval_lossValue_train',
        default=False
    )
    parser.add_argument(
        '--initialization',
        type=bool,
        help='initialization',
        default=False
    )
    parser.add_argument(
        '--num_optimizers',
        type=int,
        help='number of mapping layers from teacher',
        default=5
    )

    FLAGS, unparsed = parser.parse_known_args()
    ex = VGG16()
    tf.app.run(main=ex.main, argv=[sys.argv[0]] + unparsed)
