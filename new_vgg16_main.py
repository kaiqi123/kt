# independent student, not initialization

import tensorflow as tf
import numpy as np
import random
from DataInput import DataInput
from vgg16mentee import Mentee
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
dataset_path = "./"
tf.reset_default_graph()
NUM_ITERATIONS = 4680
SUMMARY_LOG_DIR="./summary-log"
LEARNING_RATE_DECAY_FACTOR = 0.9809
NUM_EPOCHS_PER_DECAY = 1.0
validation_accuracy_list = []
test_accuracy_list = []
seed = 1234
alpha = 0.2
random_count = 0


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
        return feed_dict

    def evaluation(self, logits, labels):
            print('evaluation')
            if FLAGS.top_1_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 1)
            elif FLAGS.top_3_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 3)
            elif FLAGS.top_5_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 5)

            return tf.reduce_sum(tf.cast(correct, tf.int32))

            ## In this function, accuracy is calculated for the training set, test set and validation set

    def do_eval(self, sess, eval_correct, logits, images_placeholder, labels_placeholder, dataset, mode, phase_train):
        true_count = 0
        if mode == 'Test':
            steps_per_epoch = FLAGS.num_testing_examples // FLAGS.batch_size
            num_examples = steps_per_epoch * FLAGS.batch_size
        if mode == 'Train':
            steps_per_epoch = FLAGS.num_training_examples // FLAGS.batch_size
            num_examples = steps_per_epoch * FLAGS.batch_size
        if mode == 'Validation':
            steps_per_epoch = FLAGS.num_validation_examples // FLAGS.batch_size
            num_examples = steps_per_epoch * FLAGS.batch_size

        for step in xrange(steps_per_epoch):
            if FLAGS.dataset == 'mnist':
                feed_dict = {images_placeholder: np.reshape(dataset.test.next_batch(FLAGS.batch_size)[0],
                                                            [FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height,
                                                             FLAGS.num_channels]),
                             labels_placeholder: dataset.test.next_batch(FLAGS.batch_size)[1]}
            else:
                feed_dict = self.fill_feed_dict(dataset, images_placeholder,
                                                labels_placeholder, sess, mode, phase_train)
            count = sess.run(eval_correct, feed_dict=feed_dict)
            true_count = true_count + count

        precision = float(true_count) / num_examples
        print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
               (num_examples, true_count, precision))
        if mode == 'Validation':
            validation_accuracy_list.append(precision)
        if mode == 'Test':
            test_accuracy_list.append(precision)

    def get_mentor_variables_to_restore(self):
        """
        Returns:: names of the weights and biases of the teacher model

        """
        return [var for var in tf.global_variables() if var.op.name.startswith("mentor") and (var.op.name.endswith("biases") or var.op.name.endswith("weights")) and (var.op.name != ("mentor_fc3/mentor_weights") and  var.op.name != ("mentor_fc3/mentor_biases"))]

    def caculate_rmse_loss(self, mentor_data_dict, mentee_data_dict):

        """
        Here layers of same width are mapped together.
        """

        ## loss between mentor's 1st layer and mentee's 1st layer
        self.l1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv1_2, mentee_data_dict.conv1_1))))

        ## loss between mentor's 4th layer and mentee's 2nd layer
        self.l2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv2_1, mentee_data_dict.conv2_1))))
        ## loss between mentor's 5th layer and mentee's 3rd layer
        self.l3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv3_1, mentee_data_dict.conv3_1))))
        ## loss between mentor's 9th layer and mentee's 4th layer
        self.l4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv4_2, mentee_data_dict.conv4_1))))
        ## loss between mentor's 12th layer and mentee's 5th layer
        self.l5 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv5_2, mentee_data_dict.conv5_1))))

    def define_multiple_optimizers(self, lr):

        print("define multiple optimizers")

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

        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(self.l1, var_list=l1_var_list)
        self.train_op2 = tf.train.AdamOptimizer(lr).minimize(self.l2, var_list=l2_var_list)
        self.train_op3 = tf.train.AdamOptimizer(lr).minimize(self.l3, var_list=l3_var_list)
        self.train_op4 = tf.train.AdamOptimizer(lr).minimize(self.l4, var_list=l4_var_list)
        self.train_op5 = tf.train.AdamOptimizer(lr).minimize(self.l5, var_list=l5_var_list)

    def train_independent_student(self, images_placeholder, labels_placeholder, seed, phase_train, global_step, sess):

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

        mentee_data_dict = student.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
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

    def train_dependent_student(self, images_placeholder, labels_placeholder, phase_train, seed, global_step, sess):

        """
        Student is trained by taking supervision from teacher for every batch of data
        Same batch of input data is passed to both teacher and student for every iteration

        """

        if FLAGS.dataset == 'cifar10' or 'mnist':
            print("Train dependent student (cifar10 or mnist)")
            vgg16_mentor = Teacher(False)
        if FLAGS.dataset == 'caltech101':
            print("Train dependent student (caltech101)")
            vgg16_mentor = Mentor(False)

        vgg16_mentee = Mentee(FLAGS.num_channels)
        self.mentor_data_dict = vgg16_mentor.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax,
                                                   phase_train)

        self.mentee_data_dict = vgg16_mentee.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed,
                                                   phase_train)
        """
        The below code is to calculate the cosine similarity between the outputs of the mentor-mentee layers.
        The layers with highest cosine similarity value are mapped together.
        
        cosine1, cosine2, cosine3, cosine4, cosine5, cosine6, cosine7, cosine8, cosine9, cosine10, cosine11, cosine12, cosine13 = self.cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict,sess)
        self.cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict,sess)
        """

        self.softmax = self.mentee_data_dict.softmax
        mentor_variables_to_restore = self.get_mentor_variables_to_restore()
        self.loss = vgg16_mentee.loss(labels_placeholder)
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # self.caculate_rmse_loss(self.mentor_data_dict, self.mentee_data_dict)
        # self.define_multiple_optimizers(lr)

        self.l1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.mentor_data_dict.conv1_2, self.mentee_data_dict.conv1_1))))
        l1_var_list = []
        l1_var_list.append([var for var in tf.global_variables() if var.op.name == "mentee_conv1_1/mentee_weights"][0])
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(self.l1, var_list=l1_var_list)

        #init = tf.constant_initializer((25,224,224,64))

        #temp = self.mentor_data_dict.conv1_2.shape
        #print(sess.run(temp))
        print(self.mentor_data_dict.conv1_2.shape)
        self.t1 = tf.Variable(tf.truncated_normal(self.mentor_data_dict.conv1_2.shape, dtype=tf.float32,
                                                 stddev=1e-2, seed=seed), name='mentor_output_layer1')
        # t1 = tf.Variable(0.0, name="mentor_output_layer1", shape = (25,224,224,64))
        # t1 = tf.get_variable('t1', shape=[25,224,224,64], initializer=init)
        self.l1_interval = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.t1, self.mentee_data_dict.conv1_1))))
        self.train_op1_interval = tf.train.AdamOptimizer(lr).minimize(self.l1_interval, var_list=l1_var_list)
        #sess.run(t1.initializer)

        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver(mentor_variables_to_restore)
        saver.restore(sess, "./summary-log/new_method_teacher_weights_filename_caltech101")


        print("initialization")
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

            if var.op.name == "mentor_fc1/mentor_weights":
                self.mentee_data_dict.parameters[10].assign(var.eval(session=sess)).eval(session=sess)

            if var.op.name == "mentor_fc3/mentor_weights":
                self.mentee_data_dict.parameters[12].assign(var.eval(session=sess)).eval(session=sess)



    def run_dependent_student(self, feed_dict, sess, i):

        if FLAGS.multiple_optimizers_l5:

            if (i % FLAGS.num_iterations == 0):
                #_, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
                self.t1.assign(sess.run(self.mentor_data_dict.conv1_2, feed_dict=feed_dict))
                #_, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
                #_, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
                #_, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)
                #_, self.loss_value5 = sess.run([self.train_op5, self.l5], feed_dict=feed_dict)

            else:
                print(i)
                #_, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
                _, self.loss_value1_interval, self.loss_value1 = sess.run([self.train_op1_interval, self.l1_interval, self.l1], feed_dict=feed_dict)
                print(self.loss_value1_interval)
                print(self.loss_value1)

    def train_model(self, data_input_train, data_input_test, images_placeholder, labels_placeholder, sess,
                    phase_train):

        try:
            print('train model')
            eval_correct = self.evaluation(self.softmax, labels_placeholder)

            for i in range(NUM_ITERATIONS):

                feed_dict = self.fill_feed_dict(data_input_train, images_placeholder,
                                                labels_placeholder, sess, 'Train', phase_train)

                if FLAGS.student or FLAGS.teacher:
                    # print("train function: independent student or teacher")
                    _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                    if i % 10 == 0:
                        print ('Step %d: loss_value = %.20f' % (i, loss_value))

                if FLAGS.dependent_student:
                    # self.normalize_the_outputs_of_mentor_mentee_of_different_widths(sess, feed_dict)
                    # self.cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict,sess,feed_dict)
                    # self.visualization_of_filters(sess)

                    self.run_dependent_student(feed_dict, sess, i)

                    if i % 1 == 0:
                        # print("train function: dependent student, multiple optimizers")
                        if FLAGS.multiple_optimizers_l5:

                            #print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            #print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            #print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                            #print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                            #print ('Step %d: loss_value5 = %.20f' % (i, self.loss_value5))
                            print ("\n")

                if (i) % (FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size) == 0 or (
                i) == NUM_ITERATIONS - 1:

                    checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
                    if FLAGS.teacher:
                        self.saver.save(sess, FLAGS.teacher_weights_filename)
                    """
                    elif FLAGS.student:
                        saver.save(sess, FLAGS.student_filename)
                    """
                    """                                            
                    elif FLAGS.dependent_student:
                        saver_new = tf.train.Saver()
                        saver_new.save(sess, FLAGS.dependent_student_filename)
                    """

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
                    print ("max accuracy % f", max(test_accuracy_list))

        except Exception as e:
            print(e)

    def main(self, _):
        start_time = time.time()
        with tf.Graph().as_default():

            print("test whether to use gpu")
            print(device_lib.list_local_devices())
            print(str(NUM_ITERATIONS))
            # This line allows the code to use only sufficient memory and does not block entire GPU
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

            # set the seed so that we have same loss values and initializations for every run.
            tf.set_random_seed(seed)

            data_input_train = DataInput(dataset_path, FLAGS.train_dataset, FLAGS.batch_size,
                                         FLAGS.num_training_examples, FLAGS.image_width, FLAGS.image_height,
                                         FLAGS.num_channels, seed, FLAGS.dataset)

            data_input_test = DataInput(dataset_path, FLAGS.test_dataset, FLAGS.batch_size, FLAGS.num_testing_examples,
                                        FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

            data_input_validation = DataInput(dataset_path, FLAGS.validation_dataset, FLAGS.batch_size,
                                              FLAGS.num_validation_examples, FLAGS.image_width, FLAGS.image_height,
                                              FLAGS.num_channels, seed, FLAGS.dataset)

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
            summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            phase_train = tf.placeholder(tf.bool, name='phase_train')
            summary = tf.summary.merge_all()

            if FLAGS.student:
                self.train_independent_student(images_placeholder, labels_placeholder, seed, phase_train, global_step,
                                               sess)

            elif FLAGS.teacher:
                self.train_teacher(images_placeholder, labels_placeholder, phase_train, global_step, sess)

            elif FLAGS.dependent_student:
                self.train_dependent_student(images_placeholder, labels_placeholder, phase_train, seed, global_step,
                                             sess)

            self.train_model(data_input_train, data_input_test, images_placeholder, labels_placeholder, sess,
                             phase_train)

            print(test_accuracy_list)
            coord.request_stop()
            coord.join(threads)

        sess.close()
        summary_writer.close()

        end_time = time.time()
        runtime = round((end_time - start_time) / (60 * 60), 2)
        print("run time is: " + str(runtime) + " hour")


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
        default="./summary-log/new_method_teacher_weights_filename_caltech101_clean_code"
    )
    parser.add_argument(
        '--student_filename',
        type=str,
        default="./summary-log/new_method_student_weights_filename_cifar10"
    )
    parser.add_argument(
        '--dependent_student_filename',
        type=str,
        default="./summary-log/new_method_dependent_student_weights_filename_cifar10"
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
        default="caltech101-train.txt"
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        default="caltech101-test.txt"
    )
    parser.add_argument(
        '--validation_dataset',
        type=str,
        default="caltech101-validation.txt"
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
        '--multiple_optimizers_l0',
        type=bool,
        help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
        default=False
    )
    parser.add_argument(
        '--multiple_optimizers_l1',
        type=bool,
        help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
        default=False
    )
    parser.add_argument(
        '--multiple_optimizers_l2',
        type=bool,
        help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
        default=False
    )
    parser.add_argument(
        '--multiple_optimizers_l3',
        type=bool,
        help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
        default=False
    )
    parser.add_argument(
        '--multiple_optimizers_l4',
        type=bool,
        help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
        default=False
    )
    parser.add_argument(
        '--multiple_optimizers_l5',
        type=bool,
        help='number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
        default=False
    )
    parser.add_argument(
        '--top_1_accuracy',
        type=bool,
        help='top-1-accuracy',
        default=True
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
        '--hard_logits',
        type=bool,
        help='hard_logits',
        default=False
    )
    parser.add_argument(
        '--fitnets_HT',
        type=bool,
        help='fitnets_HT',
        default=False
    )
    parser.add_argument(
        '--fitnets_KD',
        type=bool,
        help='fitnets_KD',
        default=False
    )
    parser.add_argument(
        '--embed_type',
        type=str,
        help='embed type can be either fully connected or convolutional layers',
        default='fc'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        help='num_iterations',
        default=1
    )
    parser.add_argument(
        '--write_accuracy',
        type=bool,
        help='write_accuracy',
        default=False
    )

    FLAGS, unparsed = parser.parse_known_args()
    ex = VGG16()
    tf.app.run(main=ex.main, argv=[sys.argv[0]] + unparsed)