import tensorflow as tf
import numpy as np
import random
from DataInput import DataInput

#from vgg16mentee import Mentee
from vgg16mentee_original import Mentee

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
#import matplotlib.pyplot as plt
dataset_path = "./"
tf.reset_default_graph()
NUM_ITERATIONS = 40
SUMMARY_LOG_DIR="./summary-log"
LEARNING_RATE_DECAY_FACTOR = 0.9809
NUM_EPOCHS_PER_DECAY = 1.0
validation_accuracy_list = []
test_accuracy_list = []
seed = 1234
alpha = 0.2
random_count = 0 



class VGG16(object):
    ### read_mnist_data is used to read mnist data. This function is not currently used as the code supports only caltech101 and CIFAR-10 for now.
    def read_mnist_data(self):
        mnist = read_data_sets(FLAGS.mnist_data_dir)
        return mnist      

    ### placeholders to hold iamges and their labels of certain datasets 
    def placeholder_inputs(self, batch_size):
            """
                Args:
                    batch_size: batch size used to train the network
                
                Returns:
                    images_placeholder: images_placeholder holds images of either caltech101 or cifar10 datasets
                    labels_placeholder: labels_placeholder holds labels of either caltech101 or cifar10 datasets

            """
            images_placeholder = tf.placeholder(tf.float32, 
                                                                    shape=(FLAGS.batch_size, FLAGS.image_height, 
                                                                               FLAGS.image_width, FLAGS.num_channels))
            labels_placeholder = tf.placeholder(tf.int32,
                                                                    shape=(FLAGS.batch_size))

            return images_placeholder, labels_placeholder

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

    ## In this function, accuracy is calculated for the training set, test set and validation set
    def do_eval(self, sess, eval_correct, logits, images_placeholder, labels_placeholder, dataset,mode, phase_train):
            true_count =0
            if mode == 'Test':
                steps_per_epoch = FLAGS.num_testing_examples //FLAGS.batch_size 
                num_examples = steps_per_epoch * FLAGS.batch_size
            if mode == 'Train':
                steps_per_epoch = FLAGS.num_training_examples //FLAGS.batch_size 
                num_examples = steps_per_epoch * FLAGS.batch_size
            if mode == 'Validation':
                steps_per_epoch = FLAGS.num_validation_examples //FLAGS.batch_size 
                num_examples = steps_per_epoch * FLAGS.batch_size

            for step in xrange(steps_per_epoch):
                if FLAGS.dataset == 'mnist':
                    feed_dict = {images_placeholder: np.reshape(dataset.test.next_batch(FLAGS.batch_size)[0], [FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels]), labels_placeholder: dataset.test.next_batch(FLAGS.batch_size)[1]}
                else:
                    feed_dict = self.fill_feed_dict(dataset, images_placeholder,
                                                            labels_placeholder,sess, mode,phase_train)
                count = sess.run(eval_correct, feed_dict=feed_dict)
                true_count = true_count + count

            precision = float(true_count) / num_examples
            print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
                            (num_examples, true_count, precision))
            if mode == 'Validation':
                validation_accuracy_list.append(precision)
            if mode == 'Test':
                test_accuracy_list.append(precision)

    def evaluation(self, logits, labels):
            print('evaluation')
            #print(logits)
            #print(FLAGS.top_1_accuracy)
            #print(FLAGS.top_3_accuracy)
            #print(FLAGS.top_5_accuracy)
            if FLAGS.top_1_accuracy: 
                correct = tf.nn.in_top_k(logits, labels, 1)
            elif FLAGS.top_3_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 3)
            elif FLAGS.top_5_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 5)

            return tf.reduce_sum(tf.cast(correct, tf.int32))

    ### while training dependent student, weights of the teacher network trained prior to the dependent student are loaded on to the teacher network to carry out inference
    def get_mentor_variables_to_restore(self):
            """
            Returns:: names of the weights and biases of the teacher model
            """
            return [var for var in tf.global_variables() if var.op.name.startswith("mentor") and (var.op.name.endswith("biases") or var.op.name.endswith("weights")) and (var.op.name != ("mentor_fc3/mentor_weights") and  var.op.name != ("mentor_fc3/mentor_biases"))]

    ### returns 1st layer weight variable of mentee network
    def l1_weights_of_mentee(self, l1_mentee_weights):
        l1_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_weights"][0])
    #    l1_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_biases"][0])

        return l1_mentee_weights

    ### returns 2nd layer weight variable of mentee network
    def l2_weights_of_mentee(self, l2_mentee_weights):
        l2_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])
     #   l2_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_biases"][0])
        return l2_mentee_weights

    ### returns 3rd layer weight variable of mentee network
    def l3_weights_of_mentee(self, l3_mentee_weights):
        l3_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
      #  l3_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_biases"][0])
        return l3_mentee_weights

    ### returns 4th layer weight variable of mentee network
    def l4_weights_of_mentee(self, l4_mentee_weights):
        l4_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv4_1/mentee_weights"][0])
       # l4_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv4_1/mentee_biases"][0])
        return l4_mentee_weights

    ### returns 5th layer weight variable of mentee network
    def l5_weights_of_mentee(self, l5_mentee_weights):
        l5_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv5_1/mentee_weights"][0])
        #l5_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv5_1/mentee_biases"][0])
        return l5_mentee_weights

    ### returns 6th layer (fully connected) weight variable of mentee network
    def l6_weights_of_mentee(self, l6_mentee_weights):
        l6_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_fc3/mentee_weights"][0])
        #l6_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_fc3/mentee_biases"][0])
        return l6_mentee_weights

    def get_variables_for_HT(self, variables_for_HT):
        
        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])

        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_weights"][0])

        return variables_for_HT
    
    def get_variables_for_KD(self, variables_for_KD):
        #return [var for var in tf.global_variables() if (var.op.name.startswith("mentee") and var.op.name.endswith("weights"))]
        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])

        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_weights"][0])
        return variables_for_KD

    def cosine_similarity_of_same_width(self, mentee_data_dict, mentor_data_dict, sess, feed_dict):

        """
            cosine similarity is calculated between 1st layer of mentee and 1st layer of mentor.
            Similarly, cosine similarity is calculated between 1st layer of mentee and 2nd layer of mentor.
        """
        normalize_a_1 = tf.nn.l2_normalize(mentee_data_dict.conv1_1,0)

        normalize_b_11 = tf.nn.l2_normalize(mentor_data_dict.conv1_1,0)        
        normalize_b_12 = tf.nn.l2_normalize(mentor_data_dict.conv1_2,0)

        """
            cosine similarity is calculated between 2th layer of mentee and 11th layer of mentor.
            Similarly, cosine similarity is calculated between 5th layer of mentee and 12th layer of mentor.
            Similarly, cosine similarity is calculated between 5th layer of mentee and 13th layer of mentor.

        """
                        
        normalize_a_2 = tf.nn.l2_normalize(mentee_data_dict.conv2_1, 0)

        normalize_b_21 = tf.nn.l2_normalize(mentor_data_dict.conv2_1,0)        
        normalize_b_22 = tf.nn.l2_normalize(mentor_data_dict.conv2_2,0)

        """
            cosine similarity is calculated between 3th layer of mentee and 8th layer of mentor.
            Similarly, cosine similarity is calculated between 4th layer of mentee and 9th layer of mentor.
            Similarly, cosine similarity is calculated between 4th layer of mentee and 10th layer of mentor.

        """
                   
        normalize_a_3 = tf.nn.l2_normalize(mentee_data_dict.conv3_1,0)

        normalize_b_31 = tf.nn.l2_normalize(mentor_data_dict.conv3_1,0)        
        normalize_b_32 = tf.nn.l2_normalize(mentor_data_dict.conv3_2,0)        
        normalize_b_33 = tf.nn.l2_normalize(mentor_data_dict.conv3_3,0)

        """
            cosine similarity is calculated between 4nd layer of mentee and 3rd layer of mentor.
            Similarly, cosine similarity is calculated between 2nd layer of mentee and 4th layer of mentor.

        """

        normalize_a_4 = tf.nn.l2_normalize(mentee_data_dict.conv4_1,0)
        normalize_a_5 = tf.nn.l2_normalize(mentee_data_dict.conv5_1,0)
        normalize_a_6 = tf.nn.l2_normalize(mentee_data_dict.conv6_1,0)

        normalize_b_41 = tf.nn.l2_normalize(mentor_data_dict.conv4_1,0)
        normalize_b_42 = tf.nn.l2_normalize(mentor_data_dict.conv4_2,0)        
        normalize_b_43 = tf.nn.l2_normalize(mentor_data_dict.conv4_3,0)
        
        normalize_b_51 = tf.nn.l2_normalize(mentor_data_dict.conv5_1,0)
        normalize_b_52 = tf.nn.l2_normalize(mentor_data_dict.conv5_2,0)        
        normalize_b_53 = tf.nn.l2_normalize(mentor_data_dict.conv5_3,0)

        """
            cosine similarity is calculated between 5rd layer of mentee and 5th layer of mentor.
            Similarly, cosine similarity is calculated between 3rd layer of mentee and 6th layer of mentor.
            Similarly, cosine similarity is calculated between 3rd layer of mentee and 7th layer of mentor.

        """
        cosine1_11=tf.reduce_sum(tf.multiply(normalize_a_1,normalize_b_11))
        cosine1_12=tf.reduce_sum(tf.multiply(normalize_a_1,normalize_b_12))

        cosine2_21=tf.reduce_sum(tf.multiply(normalize_a_2,normalize_b_21))
        cosine2_22=tf.reduce_sum(tf.multiply(normalize_a_2,normalize_b_22))

        cosine3_31=tf.reduce_sum(tf.multiply(normalize_a_3,normalize_b_31))
        cosine3_32=tf.reduce_sum(tf.multiply(normalize_a_3,normalize_b_32))
        cosine3_33=tf.reduce_sum(tf.multiply(normalize_a_3,normalize_b_33))
        
        cosine4_41=tf.reduce_sum(tf.multiply(normalize_a_4,normalize_b_41))
        cosine4_42=tf.reduce_sum(tf.multiply(normalize_a_4,normalize_b_42))
        cosine4_43=tf.reduce_sum(tf.multiply(normalize_a_4,normalize_b_43))
        #cosine4_51=tf.reduce_sum(tf.multiply(normalize_a_4,normalize_b_51))
        #cosine4_52=tf.reduce_sum(tf.multiply(normalize_a_4,normalize_b_52))
        #cosine4_53=tf.reduce_sum(tf.multiply(normalize_a_4,normalize_b_53))

        #cosine5_41=tf.reduce_sum(tf.multiply(normalize_a_5,normalize_b_41))
        #cosine5_42=tf.reduce_sum(tf.multiply(normalize_a_5,normalize_b_42))
        #cosine5_43=tf.reduce_sum(tf.multiply(normalize_a_5,normalize_b_43))
        cosine5_51=tf.reduce_sum(tf.multiply(normalize_a_5,normalize_b_51))
        cosine5_52=tf.reduce_sum(tf.multiply(normalize_a_5,normalize_b_52))
        cosine5_53=tf.reduce_sum(tf.multiply(normalize_a_5,normalize_b_53))

        #cosine6_41=tf.reduce_sum(tf.multiply(normalize_a_6,normalize_b_41))
        #cosine6_42=tf.reduce_sum(tf.multiply(normalize_a_6,normalize_b_42))
        #cosine6_43=tf.reduce_sum(tf.multiply(normalize_a_6,normalize_b_43))
        #cosine6_51=tf.reduce_sum(tf.multiply(normalize_a_6,normalize_b_51))
        #cosine6_52=tf.reduce_sum(tf.multiply(normalize_a_6,normalize_b_52))
        #cosine6_53=tf.reduce_sum(tf.multiply(normalize_a_6,normalize_b_53))

        #print(tf.multiply(normalize_a_1,normalize_b_1))
	
        print("start")
        print("1th")
        print sess.run(cosine1_11, feed_dict = feed_dict)
        print sess.run(cosine1_12, feed_dict = feed_dict)
        
        print("2th")
        print sess.run(cosine2_21, feed_dict = feed_dict)
        print sess.run(cosine2_22, feed_dict = feed_dict)
        
        print("3th")
        print sess.run(cosine3_31, feed_dict = feed_dict)
        print sess.run(cosine3_32, feed_dict = feed_dict)
        print sess.run(cosine3_33, feed_dict = feed_dict)
        
        print("4th")
        print sess.run(cosine4_41, feed_dict = feed_dict)
        print sess.run(cosine4_42, feed_dict = feed_dict)
        print sess.run(cosine4_43, feed_dict = feed_dict)
        #print sess.run(cosine4_51, feed_dict = feed_dict)
        #print sess.run(cosine4_52, feed_dict = feed_dict)
        #print sess.run(cosine4_53, feed_dict = feed_dict)
        
        print("5th")
        #print sess.run(cosine5_41, feed_dict = feed_dict)
        #print sess.run(cosine5_42, feed_dict = feed_dict)
        #print sess.run(cosine5_43, feed_dict = feed_dict)
        print sess.run(cosine5_51, feed_dict = feed_dict)
        print sess.run(cosine5_52, feed_dict = feed_dict)
        print sess.run(cosine5_53, feed_dict = feed_dict)

        #print("6th")
        #print sess.run(cosine6_41, feed_dict = feed_dict)
        #print sess.run(cosine6_42, feed_dict = feed_dict)
        #print sess.run(cosine6_43, feed_dict = feed_dict)
        #print sess.run(cosine6_51, feed_dict = feed_dict)
        #print sess.run(cosine6_52, feed_dict = feed_dict)
        #print sess.run(cosine6_53, feed_dict = feed_dict)
        print("ended")
        
        #print(cosine1_11,cosine1_12,cosine2_21,cosine2_22,cosine3_31,cosine3_32,cosine3_33,cosine4_41,cosine4_42,cosine4_43,cosine5_51,cosine5_52,cosine5_53)
        
        #return cosine1, cosine2, cosine3, cosine4, cosine5, cosine6, cosine7, cosine8, cosine9, cosine10, cosine11, cosine12, cosine13
        
    def normalize_the_outputs_of_mentor_mentee_of_different_widths(self, sess, feed_dict):

        ## normalize mentee's outputs
        normalize_mentee_1 = tf.nn.l2_normalize(self.mentee_data_dict.conv1_1, 0)
        normalize_mentee_2 = tf.nn.l2_normalize(self.mentee_data_dict.conv2_1, 0)
        normalize_mentee_3 = tf.nn.l2_normalize(self.mentee_data_dict.conv3_1, 0)
        normalize_mentee_4 = tf.nn.l2_normalize(self.mentee_data_dict.conv4_1, 0)
        normalize_mentee_5 = tf.nn.l2_normalize(self.mentee_data_dict.conv5_1, 0)

        ## normalize mentor's outputs

        normalize_mentor = {}
        normalize_mentor["1"] = tf.nn.l2_normalize(self.mentor_data_dict.conv1_1, 0)
        normalize_mentor["2"] = tf.nn.l2_normalize(self.mentor_data_dict.conv1_2, 0)
        normalize_mentor["3"] = tf.nn.l2_normalize(self.mentor_data_dict.conv2_1, 0)
        normalize_mentor["4"] = tf.nn.l2_normalize(self.mentor_data_dict.conv2_2, 0)
        normalize_mentor["5"] = tf.nn.l2_normalize(self.mentor_data_dict.conv3_1, 0)
        normalize_mentor["6"] = tf.nn.l2_normalize(self.mentor_data_dict.conv3_2, 0)
        normalize_mentor["7"] = tf.nn.l2_normalize(self.mentor_data_dict.conv3_3, 0)
        normalize_mentor["8"] = tf.nn.l2_normalize(self.mentor_data_dict.conv4_1, 0)
        normalize_mentor["9"] = tf.nn.l2_normalize(self.mentor_data_dict.conv4_2, 0)
        normalize_mentor["10"] = tf.nn.l2_normalize(self.mentor_data_dict.conv4_3, 0)
        normalize_mentor["11"] = tf.nn.l2_normalize(self.mentor_data_dict.conv5_1, 0)
        normalize_mentor["12"] = tf.nn.l2_normalize(self.mentor_data_dict.conv5_2, 0)
        normalize_mentor["13"] = tf.nn.l2_normalize(self.mentor_data_dict.conv5_3, 0)

        
        idx = tf.constant(list(xrange(0, 64)))
        self.cosine_similarity_of_different_widths(normalize_mentee_1, normalize_mentor, idx, sess, feed_dict)

        self.cosine_similarity_of_different_widths(normalize_mentee_2, normalize_mentor, idx, sess, feed_dict)

        self.cosine_similarity_of_different_widths(normalize_mentee_3, normalize_mentor, idx, sess, feed_dict)

        self.cosine_similarity_of_different_widths(normalize_mentee_4, normalize_mentor, idx, sess, feed_dict)

        self.cosine_similarity_of_different_widths(normalize_mentee_5, normalize_mentor, idx, sess, feed_dict)

    def cosine_similarity_of_different_widths(self, normalize_mentee, normalize_mentor, idx, sess, feed_dict):

        """
            cosine similarity between every layer of mentee and mentor is calculated
            normalize_mentor:: dictionary containing output of each layer of mentor
            tf.gather picks the feature maps from the mentor output such that the number of feature maps picked equals the number of feature maps of the mentee's output
            since we need to find out the similarity between the layers, a sample of feature maps from a particular layer should be suffcient rather than all the feature maps.
            thus, we use tf.gather to pick certain number of feature maps which equal number of feature maps at mentee's layer.
            idx indicate the indices of the feature maps that need to be picked form mentor's output

        """

        cosine1 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["1"], idx, axis = 3))))
        cosine2 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["2"], idx, axis = 3))))
        cosine3 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["3"], idx, axis = 3))))
        cosine4 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["4"], idx, axis = 3))))
        cosine5 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["5"], idx, axis = 3))))
        cosine6 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["6"], idx, axis = 3))))
        cosine7 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["7"], idx, axis = 3))))
        cosine8 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["8"], idx, axis = 3))))
        cosine9 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["9"], idx, axis = 3))))
        cosine10 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["10"], idx, axis = 3))))
        cosine11 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["11"], idx, axis = 3))))
        cosine12 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["12"], idx, axis = 3))))
        cosine13 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["13"], idx, axis = 3))))
        print("start")
        print sess.run(cosine1, feed_dict = feed_dict)
        print sess.run(cosine2, feed_dict = feed_dict)
        print sess.run(cosine3, feed_dict = feed_dict)
        print sess.run(cosine4, feed_dict = feed_dict)
        print sess.run(cosine5, feed_dict = feed_dict)
        print sess.run(cosine6, feed_dict = feed_dict)
        print sess.run(cosine7, feed_dict = feed_dict)
        print sess.run(cosine8, feed_dict = feed_dict)
        print sess.run(cosine9, feed_dict = feed_dict)
        print sess.run(cosine10, feed_dict = feed_dict)
        print sess.run(cosine11, feed_dict = feed_dict)
        print sess.run(cosine12, feed_dict = feed_dict)
        print sess.run(cosine13, feed_dict = feed_dict)
        print("ended")

    def loss_with_different_layer_widths(self, embed_data_dict, mentor_data_dict, mentee_data_dict):
        """
        Here layers of different widths are mapped together.

        """
        
        ## loss between the embed layers connecting mentor's 3rd layer and mentee's 1st layer
        self.l1_d = embed_data_dict.loss_embed_3
        ## loss between the embed layers connecting mentor's 5th layer and mentee's 2nd layer
        self.l2_d = embed_data_dict.loss_embed_4
        #self.l3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv3_2, mentee_data_dict.conv3_1))))
        #self.l4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv4_2, mentee_data_dict.conv4_1))))
        #self.l5 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv5_2, mentee_data_dict.conv5_1))))
        ## loss between mentor-mentee last layers before softmax
        #self.l6 = (tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.fc3l, mentee_data_dict.fc3l))))

    def visualization_of_filters(self, sess):
        
        mentor_filter = sess.run(self.mentor_data_dict.conv1_1)
        mentee_filter = sess.run(self.mentee_data_dict.conv1_1) 
        img1 = Image.fromarray(mentor_filter, 'RGB')
        img2 = Image.fromarray(mentee_filter, 'RGB')
        img1.save('mentor_filter.png')
        img2.save('mentee_filter.png')

    def rmse_loss(self, mentor_data_dict, mentee_data_dict):
        
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

        ## loss between mentor-mentee last layers before softmax
        #self.l6 = (tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.fc3l, mentee_data_dict.fc3l))))

        ## loss between mentor-mentee softmax layers
        #self.l7 = (tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.softmax, mentee_data_dict.softmax))))
        
        #ind_max = tf.argmax(mentor_data_dict.fc3l, axis = 1)
        #hard_logits = tf.one_hot(ind_max, FLAGS.num_classes)
        ## hard_logits KT technique ::: where hard_logits of teacher are transferred to student softmax output
        #self.l8 = (tf.reduce_mean(tf.square(tf.subtract(hard_logits, mentee_data_dict.softmax))))

        ## intermediate representations KT technique (single layer):: HT stands for hint based training which is phase 1 of intermediate representations KT technique.
        ## knowledge from 7th layer of mentor is given to 3rd layer of mentee.
        #self.HT = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv3_3, mentee_data_dict.conv3_1))))
    
    def calculate_loss_with_multiple_optimizers(self, feed_dict, sess):

        if FLAGS.multiple_optimizers_l0:
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
        
        elif FLAGS.multiple_optimizers_l1:
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            
            
        elif FLAGS.multiple_optimizers_l2:
           if (random_count % FLAGS.num_iterations  == 0):
            #print("mapping two layers")
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
                                    
           if (random_count % FLAGS.num_iterations  == 1):
#           print("mapping two layers different width")
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1_d = sess.run([self.train_op1_d, self.l1_d], feed_dict=feed_dict)
            _, self.loss_value2_d = sess.run([self.train_op2_d, self.l2_d], feed_dict=feed_dict)
                                    
        elif FLAGS.multiple_optimizers_l3:
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
        
        elif FLAGS.multiple_optimizers_l4:
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l5:

           if (random_count % FLAGS.num_iterations  == 0):

               _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
               global t0
               t0 = tf.convert_to_tensor(self.loss_value0, dtype=tf.float32)
               """
               _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
               _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
               _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
               _, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)
               _, self.loss_value5 = sess.run([self.train_op5, self.l5], feed_dict=feed_dict)

               
               global  mentor_preloss_list
               mentor_preloss_list = []
               mentor_preloss_list.append(tf.convert_to_tensor(sess.run(self.mentor_data_dict.conv1_2,feed_dict=feed_dict), dtype=tf.float32, name = "mentor_conv1_2_interval"))
               mentor_preloss_list.append(tf.convert_to_tensor(sess.run(self.mentor_data_dict.conv2_1,feed_dict=feed_dict), dtype=tf.float32,name = "mentor_conv2_1_interval" ))
               mentor_preloss_list.append(tf.convert_to_tensor(sess.run(self.mentor_data_dict.conv3_1,feed_dict=feed_dict), dtype=tf.float32,name = "mentor_conv3_1_interval"))
               mentor_preloss_list.append(tf.convert_to_tensor(sess.run(self.mentor_data_dict.conv4_2,feed_dict=feed_dict), dtype=tf.float32,name = "mentor_conv4_2_interval"))
               mentor_preloss_list.append(tf.convert_to_tensor(sess.run(self.mentor_data_dict.conv5_2,feed_dict=feed_dict), dtype=tf.float32,name = "mentor_conv5_2_interval"))
               print(self.mentor_data_dict.conv1_2)
               print(self.mentor_data_dict.conv2_1)
               print(self.mentor_data_dict.conv3_1)
               print(self.mentor_data_dict.conv4_2)
               print(self.mentor_data_dict.conv5_2)
               for e in mentor_preloss_list:
               print(e)
               print(len(mentor_preloss_list))
               
               global t1,t2,t3,t4,t5
               t1 = tf.convert_to_tensor(self.loss_value1, dtype=tf.float32)
               t2 = tf.convert_to_tensor(self.loss_value2, dtype=tf.float32)
               t3 = tf.convert_to_tensor(self.loss_value3, dtype=tf.float32)
               t4 = tf.convert_to_tensor(self.loss_value4, dtype=tf.float32)
               t5 = tf.convert_to_tensor(self.loss_value5, dtype=tf.float32)
               """
           else:
               #num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
               #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
               #lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
               _, self.loss_value0 = sess.run([self.train_op0_interval, t0], feed_dict=feed_dict)
               """
               _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
               _, self.loss_value1 = sess.run([self.train_op1_interval, t1], feed_dict=feed_dict)
               _, self.loss_value2 = sess.run([self.train_op2_interval, t2], feed_dict=feed_dict)
               _, self.loss_value3 = sess.run([self.train_op3_interval, t3], feed_dict=feed_dict)
               _, self.loss_value4 = sess.run([self.train_op4_interval, t4], feed_dict=feed_dict)
               _, self.loss_value5 = sess.run([self.train_op5_interval, t5], feed_dict=feed_dict)
               
	           print("random:"+str(random_count))
	           for e in mentor_preloss_list:
	           print(e)
	
               self.rmse_loss_interval(mentor_preloss_list, self.mentee_data_dict)
	           _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
               _, self.loss_value1 = sess.run([self.train_op1_interval, self.l1_interval], feed_dict=feed_dict)
               _, self.loss_value2 = sess.run([self.train_op2_interval, self.l2_interval], feed_dict=feed_dict)
               _, self.loss_value3 = sess.run([self.train_op3_interval, self.l3_interval], feed_dict=feed_dict)
               _, self.loss_value4 = sess.run([self.train_op4_interval, self.l4_interval], feed_dict=feed_dict)
               _, self.loss_value5 = sess.run([self.train_op5_interval, self.l5_interval], feed_dict=feed_dict)
	           print(self.loss_value0)
	           print(self.loss_value1)
	           print(self.loss_value2)
	           print(self.loss_value3)
	           print(self.loss_value4)
	           print(self.loss_value5)
               """
        elif FLAGS.multiple_optimizers_l6:
            if (random_count % FLAGS.num_iterations  ==0):
                _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
                _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
                _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
                _, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)
                _, self.loss_value5 = sess.run([self.train_op5, self.l5], feed_dict=feed_dict)
                _, self.loss_value6 = sess.run([self.train_op6, self.l6], feed_dict=feed_dict)

            else:
                _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            
            """
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([self.train_op5, self.l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([self.train_op6, self.l6], feed_dict=feed_dict)
            """
            

                                         
    """
        This is the traditional KT technique where all the layer weights get updated.
        if the var_list is not mentioned explicitely it means by default all the layer weights get updated

    """
    
    def calculate_loss_with_single_optimizer(self,feed_dict, sess):
        
        #if (random_count % FLAGS.num_iterations == 0):
        _, self.loss_value_all = sess.run([self.train_op_all, self.loss] , feed_dict=feed_dict)
        #    _, self.loss_value0 = sess.run([self.train_op0, self.loss] , feed_dict=feed_dict)
        #else:
        #    _, self.loss_value0 = sess.run([self.train_op0, self.loss] , feed_dict=feed_dict)
        

    """
        This is the new KT method:: where only the weights of the target layer get updated as opposed
        to traditional KT techniques where all the layer weights of the student network get updated. 
        
    """
    def train_op_for_multiple_optimizers(self, lr):

        l1_var_list = []
        l2_var_list = []
        l3_var_list = []
        l4_var_list = []
        l5_var_list = []
        l6_var_list = []
            
        print("define multiple optimizers")
        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(self.l1, var_list = self.l1_weights_of_mentee(l1_var_list))
        self.train_op2 = tf.train.AdamOptimizer(lr).minimize(self.l2, var_list = self.l2_weights_of_mentee(l2_var_list))
        self.train_op3 = tf.train.AdamOptimizer(lr).minimize(self.l3, var_list = self.l3_weights_of_mentee(l3_var_list))
        self.train_op4 = tf.train.AdamOptimizer(lr).minimize(self.l4, var_list = self.l4_weights_of_mentee(l4_var_list))
        self.train_op5 = tf.train.AdamOptimizer(lr).minimize(self.l5, var_list = self.l5_weights_of_mentee(l5_var_list))
        self.train_op6 = tf.train.AdamOptimizer(lr).minimize(self.l6, var_list = self.l6_weights_of_mentee(l6_var_list))
            
#       self.train_op1_d = tf.train.AdamOptimizer(lr).minimize(self.l1_d, var_list = self.l1_weights_of_mentee(l1_var_list))
#       self.train_op2_d = tf.train.AdamOptimizer(lr).minimize(self.l2_d, var_list = self.l2_weights_of_mentee(l2_var_list))


    def train_op_for_multiple_optimizers_interval(self, lr):

        l1_var_list = []
        l2_var_list = []
        l3_var_list = []
        l4_var_list = []
        l5_var_list = []
        l6_var_list = []

        """
        self.l1_interval = self.l1
        self.l2_interval = self.l2
        self.l3_interval = self.l3
        self.l4_interval = self.l4
        self.l5_interval = self.l5
    
        print("define multiple optimizers interval")
        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.train_op1_interval = tf.train.AdamOptimizer(lr).minimize(self.l1_interval, var_list = self.l1_weights_of_mentee(l1_var_list))
        self.train_op2_interval = tf.train.AdamOptimizer(lr).minimize(self.l2_interval, var_list = self.l2_weights_of_mentee(l2_var_list))
        self.train_op3_interval = tf.train.AdamOptimizer(lr).minimize(self.l3_interval, var_list = self.l3_weights_of_mentee(l3_var_list))
        self.train_op4_interval = tf.train.AdamOptimizer(lr).minimize(self.l4_interval, var_list = self.l4_weights_of_mentee(l4_var_list))
        self.train_op5_interval = tf.train.AdamOptimizer(lr).minimize(self.l5_interval, var_list = self.l5_weights_of_mentee(l5_var_list))
           
        t1 = tf.convert_to_tensor(4.0, dtype=tf.float32)
        t2 = tf.convert_to_tensor(0.6, dtype=tf.float32)
        t3 = tf.convert_to_tensor(0.6, dtype=tf.float32)
        t4 = tf.convert_to_tensor(0.6, dtype=tf.float32)
        t5 = tf.convert_to_tensor(0.6, dtype=tf.float32)
        print(t1)
        """
        t1 = self.l1
        t2 = self.l2
        t3 = self.l3
        t4 = self.l4
        t5 = self.l5
        print("define multiple optimizers interval")
        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.train_op1_interval = tf.train.AdamOptimizer(lr).minimize(t1, var_list = self.l1_weights_of_mentee(l1_var_list))
        self.train_op2_interval = tf.train.AdamOptimizer(lr).minimize(t2, var_list = self.l2_weights_of_mentee(l2_var_list))
        self.train_op3_interval = tf.train.AdamOptimizer(lr).minimize(t3, var_list = self.l3_weights_of_mentee(l3_var_list))
        self.train_op4_interval = tf.train.AdamOptimizer(lr).minimize(t4, var_list = self.l4_weights_of_mentee(l4_var_list))
        self.train_op5_interval = tf.train.AdamOptimizer(lr).minimize(t5, var_list = self.l5_weights_of_mentee(l5_var_list))

    def train_op_for_single_optimizer(self, lr):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        if FLAGS.single_optimizer_l1:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1)
        elif FLAGS.single_optimizer_l2:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2)
        elif FLAGS.single_optimizer_l3:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3)
        elif FLAGS.single_optimizer_l4:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3 + self.l4)
        
        elif FLAGS.single_optimizer_l5:
            
            print("single optimizer l5")
            #print(FLAGS.add_weight)           
            #if FLAGS.add_weight:
               #print("single optimizer: add weight")
               #self.train_op_all = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3 + self.l4 + self.l5)
            #else:
               #print("single optimizer:no add weight")
            self.train_op_all = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3 + self.l4 + self.l5)
            #self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)               
                
        elif FLAGS.single_optimizer_l6:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3 + self.l4 + self.l5 + self.l6)
        elif FLAGS.single_optimizer_last_layer:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l6)

        #####Hard Logits KT technique
        elif FLAGS.hard_logits:

            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.l8)
        
        #####Soft Logits KT technique
        elif FLAGS.single_optimizer_last_layer_with_temp_softmax:
            print('single optomizer:soft logits:self.loss+self.l7')
            self.train_op = tf.train.AdamOptimizer(lr).minimize(alpha * self.loss + self.l7)

        #####Phase 1 of intermediate representations KT technique
        elif FLAGS.fitnets_HT:
            print("single optimizer: HT")
            variables_for_HT = []
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.HT, var_list = self.get_variables_for_HT(variables_for_HT))

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
        
        mentee_data_dict = student.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax,seed, phase_train)
        self.loss = student.loss(labels_placeholder)
        ## learning rate is decayed exponentially with a decay factor of 0.9809 after every epoch
        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
        self.train_op = student.training(self.loss,lr, global_step)
        self.softmax = mentee_data_dict.softmax
        # initialize all the variables of the network
        init = tf.initialize_all_variables()
        sess.run(init)
        ## saver object is created to save all the variables to a file
        self.saver = tf.train.Saver()

    def train_teacher(self, images_placeholder, labels_placeholder, phase_train, global_step, sess):

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
        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
        if FLAGS.dataset == 'caltech101':
            ## restore all the weights 
            variables_to_restore = self.get_mentor_variables_to_restore()
            self.train_op = mentor.training(self.loss, FLAGS.learning_rate_pretrained,lr, global_step, variables_to_restore,mentor.get_training_vars())
        if FLAGS.dataset == 'cifar10':
            self.train_op = mentor.training(self.loss, FLAGS.learning_rate, global_step)
        self.softmax = mentor_data_dict.softmax
        init = tf.global_variables_initializer()
        sess.run(init)
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
        self.mentor_data_dict = vgg16_mentor.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, phase_train)

        self.mentee_data_dict = vgg16_mentee.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)
        """
        The below code is to calculate the cosine similarity between the outputs of the mentor-mentee layers.
        The layers with highest cosine similarity value are mapped together.
        """
        #cosine1, cosine2, cosine3, cosine4, cosine5, cosine6, cosine7, cosine8, cosine9, cosine10, cosine11, cosine12, cosine13 = self.cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict,sess)
        
        #self.cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict,sess)
        
        self.softmax = self.mentee_data_dict.softmax
        mentor_variables_to_restore = self.get_mentor_variables_to_restore()
        self.loss = vgg16_mentee.loss(labels_placeholder)
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
        
        if FLAGS.single_optimizer and FLAGS.layers_with_same_width:
            print("single_optimizer and layers_with_same_width")
            if (random_count % FLAGS.num_iterations == 0):
                self.rmse_loss(self.mentor_data_dict, self.mentee_data_dict)
            self.train_op_for_single_optimizer(lr)
            init = tf.initialize_all_variables()
            sess.run(init)

        elif FLAGS.single_optimizer and FLAGS.layers_with_different_widths:
            print("single_optimizer and layers_with_different_widths")
            embed = Embed()
            embed_data_dict  = embed.build(self.mentor_data_dict, self.mentee_data_dict, FLAGS.embed_type)
            self.loss_with_different_layer_widths(embed_data_dict, self.mentor_data_dict, self.mentee_data_dict)
            self.train_op_for_single_optimizer(lr)
            init = tf.initialize_all_variables()
            sess.run(init)

        elif FLAGS.multiple_optimizers and FLAGS.layers_with_same_width:
            print("multiple_optimizers and layers_with_same_width")
            #self.rmse_loss(self.mentor_data_dict, self.mentee_data_dict)
            #self.train_op_for_multiple_optimizers(lr)
            #self.train_op_for_multiple_optimizers_interval(lr)
            self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)
            self.train_op0_interval = tf.train.AdamOptimizer(lr).minimize(t0)

            init = tf.initialize_all_variables()
            sess.run(init)

        elif FLAGS.multiple_optimizers and FLAGS.layers_with_different_widths:
            print("multiple_optimizers and layers_with_different_widths")
            embed = Embed()
            embed_data_dict  = embed.build(self.mentor_data_dict, self.mentee_data_dict, FLAGS.embed_type)
            self.loss_with_different_layer_widths(embed_data_dict, self.mentor_data_dict, self.mentee_data_dict)
            self.train_op_for_multiple_optimizers(lr)
            init = tf.initialize_all_variables()
            sess.run(init)
        elif FLAGS.fitnets_KD:
            print("fitness KD")
            init = tf.initialize_all_variables()
            sess.run(init)
            variables_for_KD = []
            saver = tf.train.Saver(self.get_variables_for_KD(variables_for_KD))
            saver.restore(sess, "./summary-log/new_method_dependent_student_weights_filename_cifar10")
        
        saver = tf.train.Saver(mentor_variables_to_restore)
        saver.restore(sess, "./summary-log/new_method_teacher_weights_filename_caltech101")

        """
        print("initialization")
        for var in tf.global_variables():
            if var.op.name=="mentor_conv1_1/mentor_weights":
                         self.mentee_data_dict.parameters[0].assign(var.eval(session = sess)).eval(session = sess)
            
                if var.op.name=="mentor_conv2_1/mentor_weights":
                         self.mentee_data_dict.parameters[2].assign(var.eval(session = sess)).eval(session = sess)
           
                if var.op.name=="mentor_conv3_1/mentor_weights":
                         self.mentee_data_dict.parameters[4].assign(var.eval(session = sess)).eval(session = sess)
           
                if var.op.name=="mentor_conv4_1/mentor_weights":
                         self.mentee_data_dict.parameters[6].assign(var.eval(session = sess)).eval(session = sess)
           
                if var.op.name=="mentor_conv5_1/mentor_weights":
                         self.mentee_data_dict.parameters[8].assign(var.eval(session = sess)).eval(session = sess)
           
                    if var.op.name=="mentor_fc1/mentor_weights":
                         self.mentee_data_dict.parameters[10].assign(var.eval(session = sess)).eval(session = sess)
           
                if var.op.name=="mentor_fc3/mentor_weights":
                         self.mentee_data_dict.parameters[12].assign(var.eval(session = sess)).eval(session = sess)
           
            print("222")
            print(sess.run(self.mentee_data_dict.parameters[0][0][0][0][0]))
            print(sess.run(self.mentee_data_dict.parameters[2][0][0][0][0]))
            print(sess.run(self.mentee_data_dict.parameters[4][0][0][0][0]))
            print(sess.run(self.mentee_data_dict.parameters[6][0][0][0][0]))
            print(sess.run(self.mentee_data_dict.parameters[8][0][0][0][0]))
            print(sess.run(self.mentee_data_dict.parameters[14][0]))
            print(sess.run(self.mentee_data_dict.parameters[16][0]))
        for var in tf.global_variables():
            if var.op.name=="mentee_conv1_1/mentee_weights":
                     print(sess.run(var[0][0][0][0]))
    
            if var.op.name=="mentee_conv2_1/mentee_weights":
                     print(sess.run(var[0][0][0][0]))
    
            if var.op.name=="mentee_conv3_1/mentee_weights":
                     print(sess.run(var[0][0][0][0]))
    
            if var.op.name=="mentee_conv4_1/mentee_weights":
                     print(sess.run(var[0][0][0][0]))
    
            if var.op.name=="mentee_conv5_1/mentee_weights":
                     print(sess.run(var[0][0][0][0]))
    
            if var.op.name=="mentee_fc2/mentee_weights":
                     print(sess.run(var[0]))
    
            if var.op.name=="mentee_fc3/mentee_weights":
                     print(sess.run(var[0]))
             """

    def train_model(self, data_input_train, data_input_test, images_placeholder, labels_placeholder, sess, phase_train):
        
        try:
            print('train model')
            #print(self.softmax)
            #print(labels_placeholder)
            #print('444')
            eval_correct= self.evaluation(self.softmax, labels_placeholder)
            #count = 0
            #print("222")
            #print(eval_correct)
            for i in range(NUM_ITERATIONS):
#                print("i:"+str(i))
#                start_time = time.time()
                global random_count
                
                feed_dict = self.fill_feed_dict(data_input_train, images_placeholder,
                                                                    labels_placeholder, sess, 'Train', phase_train)
                #print(FLAGS)
                if FLAGS.student or FLAGS.teacher:
                    #print("train function: independent student or teacher")
                    _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if FLAGS.dataset == 'mnist':
                        batch = mnist.train.next_batch(FLAGS.batch_size)
                        _, loss_value = sess.run([self.train_op, self.loss], feed_dict = {images_placeholder: np.reshape(batch[0], [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels]), labels_placeholder: batch[1]})

                    if i % 10 == 0:
                        print ('Step %d: loss_value = %.20f' % (i, loss_value))

                if FLAGS.dependent_student and FLAGS.single_optimizer:

                    self.calculate_loss_with_single_optimizer(feed_dict, sess)
                    if i % 1 == 0:
                        #print("train function: dependent student, single optimizer")
                        #if random_count % FLAGS.num_iterations == 0:
                        print ('Step %d: loss_value_all = %.20f' % (i, self.loss_value_all))
                           #print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                        #else:
                        #   print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))

                if FLAGS.dependent_student and FLAGS.multiple_optimizers:
                    #self.normalize_the_outputs_of_mentor_mentee_of_different_widths(sess, feed_dict)
                    #self.cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict,sess,feed_dict)
                    #self.visualization_of_filters(sess)
                    

                    self.calculate_loss_with_multiple_optimizers(feed_dict, sess)
                                                                
                    if i % 10 == 0:
                        #print("train function: dependent student, multiple optimizers")
                        if FLAGS.multiple_optimizers_l0:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                        elif FLAGS.multiple_optimizers_l1:

                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))

                        elif FLAGS.multiple_optimizers_l2:
                            if (random_count % FLAGS.num_iterations  == 0):
                                print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                                print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                                print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            if (random_count % FLAGS.num_iterations  == 1):
                                print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                                print ('Step %d: loss_value1_d = %.20f' % (i, self.loss_value1_d))
                                print ('Step %d: loss_value2_d = %.20f' % (i, self.loss_value2_d))
                        elif FLAGS.multiple_optimizers_l3:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                        elif FLAGS.multiple_optimizers_l4:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                            print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                        elif FLAGS.multiple_optimizers_l5:

                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            """
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                            print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                            print ('Step %d: loss_value5 = %.20f' % (i, self.loss_value5))
                            print ("\n")
                            """

                        elif FLAGS.multiple_optimizers_l6:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                            print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                            print ('Step %d: loss_value5 = %.20f' % (i, self.loss_value5))
                            print ('Step %d: loss_value6 = %.20f' % (i, self.loss_value6))

                                                
                        """
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i)
                        summary_writer.flush()
                        """
                    if (i) %(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//FLAGS.batch_size)  == 0 or (i) == NUM_ITERATIONS-1:

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

                        if FLAGS.dataset == 'mnist':
                            print('validation accuracy::MNIST')
                            self.do_eval(sess,
                                eval_correct,
                                softmax,
                                images_placeholder,
                                labels_placeholder,
                                mnist,
                                'Validation', phase_train)

                            print('test accuracy::MNIST')
                            self.do_eval(sess,
                                eval_correct,
                                softmax,
                                images_placeholder,
                                labels_placeholder,
                                mnist,
                                'Test', phase_train)

                        else:
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
                            #print ("test accuracy", test_accuracy_list)
                    random_count = random_count + 1
    #                print( "--- %s seconds ---" % (time.time() - start_time))
        except Exception as e:
            print(e)
    
    
    def main(self, _):
            start_time = time.time()
            with tf.Graph().as_default():
                    
                print("test whether to use gpu")
                print(device_lib.list_local_devices())
                print(str(NUM_ITERATIONS))
                ## This line allows the code to use only sufficient memory and does not block entire GPU
                config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
                #print('11111'+FLAGS.dataset)
                if FLAGS.dataset == 'mnist':
                    print('dataset is mnist')
                    mnist = read_mnist_data()

                ## set the seed so that we have same loss values and initializations for every run.
                tf.set_random_seed(seed)

                data_input_train = DataInput(dataset_path, FLAGS.train_dataset, FLAGS.batch_size, FLAGS.num_training_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

                data_input_test = DataInput(dataset_path, FLAGS.test_dataset, FLAGS.batch_size, FLAGS.num_testing_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

                data_input_validation = DataInput(dataset_path, FLAGS.validation_dataset,FLAGS.batch_size, FLAGS.num_validation_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)
                images_placeholder, labels_placeholder = self.placeholder_inputs(FLAGS.batch_size)

                #config = tf.ConfigProto(allow_soft_placement=True)
                config = tf.ConfigProto()
                config.gpu_options.allocator_type = 'BFC'
                #config.gpu_options.per_process_gpu_memory_fraction = 0.90

                sess = tf.Session(config = config)
                ## this line is used to enable tensorboard debugger
                #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
                summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                global global_step
                global_step = tf.Variable(0, name='global_step', trainable=False)
                phase_train = tf.placeholder(tf.bool, name = 'phase_train')
                summary = tf.summary.merge_all()

                if FLAGS.student:
                    self.train_independent_student(images_placeholder, labels_placeholder, seed, phase_train, global_step, sess)

                elif FLAGS.teacher:
                    self.train_teacher(images_placeholder, labels_placeholder, phase_train, global_step, sess)

                elif FLAGS.dependent_student:
                    self.train_dependent_student(images_placeholder, labels_placeholder, phase_train, seed, global_step, sess)

                self.train_model(data_input_train, data_input_test, images_placeholder, labels_placeholder, sess, phase_train)
                """
                if FLAGS.dependent_student:
                   if FLAGS.multiple_optimizers:		
                        file1="output/dependent_student/multiple_optimizers/"

                        if FLAGS.multiple_optimizers_l5:
                            print("write to multiple_optimizers_l5")
                            openfile = open(file1 + "5layers/" + "addWeight"+str(FLAGS.add_weight)+"_"+str(FLAGS.num_iterations) + ".csv", 'wb')

                        if FLAGS.multiple_optimizers_l6:
                            print("write to multiple_optimizers_l6")
                            openfile = open(file1 + "6layers/" + str(FLAGS.num_iterations) + ".csv", 'wb')

                        if FLAGS.multiple_optimizers_l4:
                            print("write to multiple_optimizers_l4")
                            openfile = open(file1 + "4layers/" + str(FLAGS.num_iterations) + ".csv", 'wb')

                        if FLAGS.multiple_optimizers_l3:
                            print("write to multiple_optimizers_l3")
                            openfile = open(file1 + "3layers/" + str(FLAGS.num_iterations) + ".csv", 'wb')

                        if FLAGS.multiple_optimizers_l2:
                            print("write to multiple_optimizers_l2")
                            openfile = open(file1 + "2layers/" + str(FLAGS.num_iterations) + ".csv", 'wb')

                        if FLAGS.multiple_optimizers_l1:
                            print("write to multiple_optimizers_l1")
                            openfile = open(file1 + "1layers/" + str(FLAGS.num_iterations) + ".csv", 'wb')

                   if FLAGS.single_optimizer:
                        file2 = "output/dependent_student/single_optimizer/"

                        if FLAGS.single_optimizer_l5:
                           print("write to single_optimizer_l5")
                           openfile = open(file2 + "5layers/" + "addWeight"+str(FLAGS.add_weight)+"_"+str(FLAGS.num_iterations) + ".csv", 'wb')


                   wr = csv.writer(openfile, dialect='excel')
                   print(test_accuracy_list)
                   wr.writerow(test_accuracy_list)
                
                if FLAGS.teacher:
                   print("write to teacher_accuracy.csv")
                   openfile = open("output/teacher/teacher_accuracy_std1102.csv",'wb')
                   wr = csv.writer(openfile, dialect='excel')
                   print(test_accuracy_list)
                   wr.writerow(test_accuracy_list)
                """
                print(test_accuracy_list)
                coord.request_stop()
                coord.join(threads)

            sess.close()
            summary_writer.close()
	    
            end_time = time.time()
            runtime  =  round((end_time - start_time)/(60*60),2)
            print("run time is: "+str(runtime)+" hour")

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--teacher',
            type = bool,
            help = 'train teacher',
            default = False
        )
        parser.add_argument(
                '--dependent_student',
                type = bool,
                help = 'train dependent student',
                default = False
            )
        parser.add_argument(
            '--student',
            type = bool,
            help = 'train independent student',
            default = False
        )
        parser.add_argument(
            '--teacher_weights_filename',
            type = str,
            default = "./summary-log/new_method_teacher_weights_filename_caltech101_clean_code"
        )
        parser.add_argument(
            '--student_filename',
            type = str,
            default = "./summary-log/new_method_student_weights_filename_cifar10"
        )
        parser.add_argument(
            '--dependent_student_filename',
            type = str,
            default = "./summary-log/new_method_dependent_student_weights_filename_cifar10"
        )

        parser.add_argument(
            '--learning_rate',
            type = float,
            default = 0.0001
        )

        parser.add_argument(
            '--batch_size',
            type = int,
            default = 25                                   
        )
        parser.add_argument(
            '--image_height',
            type = int,
            default = 224                                   
        )
        parser.add_argument(
            '--image_width',
            type = int,
            default = 224                                  
        )
        parser.add_argument(
            '--train_dataset',
            type = str,
            default = "caltech101-train.txt"                                   
        )
        parser.add_argument(
            '--test_dataset',
            type = str,
            default = "caltech101-test.txt"                                   
        )
        parser.add_argument(
            '--validation_dataset',
            type = str,
            default = "caltech101-validation.txt"                                   
        )
        parser.add_argument(
            '--temp_softmax',
            type = int,
            default = 1                                   
        )
        parser.add_argument(
            '--num_classes',
            type = int,
            default = 102                                   
        )
        parser.add_argument(
            '--learning_rate_pretrained',
            type = float,
            default = 0.0001                                   
        )
        parser.add_argument(
            '--NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN',
            type = int,
            default = 5853                                    
        )
        parser.add_argument(
            '--num_training_examples',
            type = int,
            default = 5853
        )
        parser.add_argument(
            '--num_testing_examples',
            type = int,
            default = 1829                                    
        )
        parser.add_argument(
            '--num_validation_examples',
            type = int,
            default = 1463                                    
        )
        parser.add_argument(
            '--single_optimizer',
            type = bool,
            help = 'single_optimizer',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers',
            type = bool,
            help = 'multiple_optimizers',
            default = False
        )
        parser.add_argument(
            '--dataset',
            type = str,
            help = 'name of the dataset',
            default = 'caltech101'
        )
        parser.add_argument(
            '--mnist_data_dir',
            type = str,
            help = 'name of the dataset',
            default = './mnist_data'
        )
        parser.add_argument(
            '--num_channels',
            type = int,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = '3'
        )
        parser.add_argument(
            '--single_optimizer_l1',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l2',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l3',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l4',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l5',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l6',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_last_layer',
            type = bool,
            help = 'last layer loss from mentor only the logits',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_last_layer_with_temp_softmax',
            type = bool,
            help = 'last layer loss from mentor with temp softmax',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l0',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default =  False
        )
        parser.add_argument(
            '--multiple_optimizers_l1',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l2',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default =  False
        )
        parser.add_argument(
            '--multiple_optimizers_l3',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l4',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l5',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l6',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--top_1_accuracy',
            type = bool,
            help = 'top-1-accuracy',
            default = True
        )
        parser.add_argument(
            '--top_3_accuracy',
            type = bool,
            help = 'top-3-accuracy',
            default = False
        )
        parser.add_argument(
            '--top_5_accuracy',
            type = bool,
            help = 'top-5-accuracy',
            default = False
        )
        parser.add_argument(
            '--hard_logits',
            type = bool,
            help = 'hard_logits',
            default = False
        )
        parser.add_argument(
            '--fitnets_HT',
            type = bool,
            help = 'fitnets_HT',
            default = False
        )
        parser.add_argument(
            '--fitnets_KD',
            type = bool,
            help = 'fitnets_KD',
            default = False
        )
        parser.add_argument(
            '--layers_with_different_widths',
            type = bool,
            help = 'different width layers mapping',
            default = False 
        )
        parser.add_argument(
            '--embed_type',
            type = str,
            help = 'embed type can be either fully connected or convolutional layers',
            default = 'fc'
        )
        parser.add_argument(
            '--layers_with_same_width',
            type = bool,
            help = 'same width layers mapping',
            default = False
        )
        parser.add_argument(
            '--num_iterations',
            type = int,
            help = 'num_iterations',
            default = 1
        )
        parser.add_argument(
            '--write_accuracy',
            type = bool,
            help = 'write_accuracy',
            default = False
        )
        parser.add_argument(
            '--add_weight',
            type = bool,
            help = 'add_weight',
            default = False
        )

        
        FLAGS, unparsed = parser.parse_known_args()
        ex = VGG16()
        tf.app.run(main=ex.main, argv = [sys.argv[0]] + unparsed)


