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
NUM_ITERATIONS = 11700
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
        #student = Mentee(FLAGS.num_channels)

        if FLAGS.num_optimizers == 6:
            #mentee_data_dict = student.build_conv6fc3(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed)
            mentee_data_dict = student.build_student_conv6fc3(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        elif FLAGS.num_optimizers == 5:
            mentee_data_dict = student.build_student_conv5fc2(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        elif FLAGS.num_optimizers == 4:
            mentee_data_dict = student.build_student_conv4fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        elif FLAGS.num_optimizers == 3:
            mentee_data_dict = student.build_student_conv3fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        elif FLAGS.num_optimizers == 2:
            mentee_data_dict = student.build_student_conv2fc1(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        else:
            raise ValueError("Not found num_optimizers")

        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,staircase=True)

        self.loss = student.loss(labels_placeholder)
        self.train_op = student.training(self.loss, lr, global_step)
        self.softmax = mentee_data_dict.softmax

        for tvar in tf.trainable_variables():
            print(tvar)
        print('Mentee, trainable variables: %d' % len(tf.trainable_variables()))

        student._calc_num_trainable_params()
        init = tf.initialize_all_variables()
        sess.run(init)
        self.saver = tf.train.Saver()

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
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,staircase=True)

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
            self.train_op = mentor.training(self.loss, FLAGS.learning_rate_pretrained, lr, global_step, variables_to_restore, mentor.get_training_vars())
        elif FLAGS.dataset == 'cifar10':
            self.train_op = mentor.training(self.loss, lr, global_step)
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

                if var.op.name == "mentor/fc1/weights":
                    print("initialization: fc1 weights")
                    self.mentee_data_dict.parameters[10].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/fc1/biases":
                    print("initialization: fc1 biases")
                    self.mentee_data_dict.parameters[11].assign(var.eval(session=sess)).eval(session=sess)

                if var.op.name == "mentor/fc3/weights":
                    print("initialization: fc3 weights")
                    self.mentee_data_dict.parameters[12].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/fc3/biases":
                    print("initialization: fc3 biases")
                    self.mentee_data_dict.parameters[13].assign(var.eval(session=sess)).eval(session=sess)

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

                if var.op.name == "mentor/fc3/weights":
                    print("initialization: fc3 weights")
                    self.mentee_data_dict.parameters[16].assign(var.eval(session=sess)).eval(session=sess)
                if var.op.name == "mentor/fc3/biases":
                    print("initialization: fc3 biases")
                    self.mentee_data_dict.parameters[17].assign(var.eval(session=sess)).eval(session=sess)

    def caculate_rmse_loss(self):

        def build_loss(teacher_layer, student_layer):
            #norm_teacher = tf.nn.l2_normalize(teacher_layer, axis=0)
            #norm_student = tf.nn.l2_normalize(student_layer, axis=0)
            loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(teacher_layer, student_layer))))
            #loss_layer = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(norm_teacher, norm_student))))
            return loss_layer

        #self.loss_softmax = build_loss(self.mentor_data_dict.softmax, self.mentee_data_dict.softmax)
        self.l1 = build_loss(self.mentor_data_dict.conv1_2, self.mentee_data_dict.conv1_1)
        if FLAGS.num_optimizers >= 2:
            self.l2 = build_loss(self.mentor_data_dict.conv2_1, self.mentee_data_dict.conv2_1)
        if FLAGS.num_optimizers >= 3:
            self.l3 = build_loss(self.mentor_data_dict.conv3_1, self.mentee_data_dict.conv3_1)
        if FLAGS.num_optimizers >= 4:
            self.l4 = build_loss(self.mentor_data_dict.conv4_2, self.mentee_data_dict.conv4_1)
        if FLAGS.num_optimizers == 5:
            self.l5 = build_loss(self.mentor_data_dict.conv5_2, self.mentee_data_dict.conv5_1)

    def define_multiple_optimizers(self, lr):

        print("define multiple optimizers")
        tvars = [var for var in tf.trainable_variables() if var.op.name.startswith("mentee")]
        #self.train_op_softmax = tf.train.AdamOptimizer(lr).minimize(self.loss_softmax, var_list=tvars)
        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=tvars)
        for var in tvars:
            print(var)
        print('num of mentee trainable_variables: %d' % len(tvars))

        l1_var_list = [var for var in tf.trainable_variables() if var.op.name == "mentee/conv1_1/weights"
                       or var.op.name == "mentee/conv1_1/biases"]
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(self.l1, var_list=l1_var_list)
        print(l1_var_list)

        if FLAGS.num_optimizers >= 2:
            l2_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/conv2_1/weights"
                           or var.op.name == "mentee/conv2_1/biases"]
            self.train_op2 = tf.train.AdamOptimizer(lr).minimize(self.l2, var_list=l2_var_list)
            print(l2_var_list)

        if FLAGS.num_optimizers >= 3:
            l3_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/conv3_1/weights"
                           or var.op.name == "mentee/conv3_1/biases"]
            self.train_op3 = tf.train.AdamOptimizer(lr).minimize(self.l3, var_list=l3_var_list)
            print(l3_var_list)

        if FLAGS.num_optimizers >= 4:
            l4_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/conv4_1/weights"
                           or var.op.name == "mentee/conv4_1/biases"]
            self.train_op4 = tf.train.AdamOptimizer(lr).minimize(self.l4, var_list=l4_var_list)
            print(l4_var_list)

        if FLAGS.num_optimizers == 5:
            l5_var_list = [var for var in tf.trainable_variables() if var.op.name=="mentee/conv5_1/weights"
                           or var.op.name == "mentee/conv5_1/biases"]
            self.train_op5 = tf.train.AdamOptimizer(lr).minimize(self.l5, var_list=l5_var_list)
            print(l5_var_list)

    def define_dependent_student(self, images_placeholder, labels_placeholder, seed, global_step, sess):
        if FLAGS.dataset == 'cifar10':
            print("Build dependent student (cifar10)")
            vgg16_mentor = TeacherForCifar10(False)
        elif FLAGS.dataset == 'caltech101':
            print("Build dependent student (caltech101)")
            vgg16_mentor = MentorForCaltech101(False)
        else:
            raise ValueError("Not found dataset name")

        vgg16_mentee = Mentee(seed)
        self.mentor_data_dict = vgg16_mentor.build_vgg16_teacher(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)

        if FLAGS.num_optimizers == 6:
            self.mentee_data_dict = vgg16_mentee.build_student_conv6fc3(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
        elif FLAGS.num_optimizers == 5:
            self.mentee_data_dict = vgg16_mentee.build_student_conv5fc2(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax)
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
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

        self.caculate_rmse_loss()
        self.define_multiple_optimizers(lr)
        vgg16_mentee._calc_num_trainable_params()

        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver(mentor_variables_to_restore)
        saver.restore(sess, FLAGS.teacher_weights_filename)

        if FLAGS.initialization:
            self.initilize(sess)

    def run_dependent_student(self, feed_dict, sess, i):

        if (i % FLAGS.num_iterations == 0):

            #print("connect teacher: "+str(i))

            #self.cosine = cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict, sess, feed_dict, FLAGS.num_optimizers)
            #cosine = sess.run(self.cosine, feed_dict=feed_dict)
            #self.select_optimizers_and_loss(cosine)

            if FLAGS.num_optimizers == 6:
                _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
                _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
                _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
                # _, self.loss_value_soft = sess.run([self.train_op_soft, self.softloss], feed_dict=feed_dict)
            else:
                #_, self.loss_value_soft = sess.run([self.train_op_soft, self.softloss], feed_dict=feed_dict)
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

        else:
            #print("do not connect teacher: "+str(i))
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)

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

    def compute_0filter_of_teacherOutput(self, sess, feed_dict):
        mentor_conv1_1, mentor_conv1_2, \
        mentor_conv2_1, mentor_conv2_2, \
        mentor_conv3_1, mentor_conv3_2, mentor_conv3_3, \
        mentor_conv4_1, mentor_conv4_2, mentor_conv4_3, \
        mentor_conv5_1, mentor_conv5_2, mentor_conv5_3, \
        mentor_fc1, mentor_fc2 \
            = sess.run([self.mentor_data_dict.conv1_1, self.mentor_data_dict.conv1_2,
                        self.mentor_data_dict.conv2_1, self.mentor_data_dict.conv2_2,
                        self.mentor_data_dict.conv3_1, self.mentor_data_dict.conv3_2, self.mentor_data_dict.conv3_3,
                        self.mentor_data_dict.conv4_1, self.mentor_data_dict.conv4_2, self.mentor_data_dict.conv4_3,
                        self.mentor_data_dict.conv5_1, self.mentor_data_dict.conv5_2, self.mentor_data_dict.conv5_3,
                        self.mentor_data_dict.fc1, self.mentor_data_dict.fc2],
                       feed_dict=feed_dict)
        self.count_filter0_num(mentor_conv1_1, "conv1_1")
        self.count_filter0_num(mentor_conv1_2, "conv1_2")
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

        np.save("output_vgg16/filters_npy/mentor_conv1_1.npy", mentor_conv1_1)
        np.save("output_vgg16/filters_npy/mentor_conv2_1.npy", mentor_conv2_1)
        np.save("output_vgg16/filters_npy/mentor_conv3_1.npy", mentor_conv3_1)
        np.save("output_vgg16/filters_npy/mentor_conv4_1.npy", mentor_conv4_1)
        np.save("output_vgg16/filters_npy/mentor_conv5_1.npy", mentor_conv5_1)

    def train_model(self, data_input_train, data_input_test, images_placeholder, labels_placeholder, sess):

        try:
            print('Begin to train model.....................................................')

            eval_correct = self.evaluation(self.softmax, labels_placeholder)

            for i in range(NUM_ITERATIONS):

                feed_dict, images_feed, labels_feed = self.fill_feed_dict(data_input_train, images_placeholder, labels_placeholder, sess)

                if FLAGS.student or FLAGS.teacher:

                    _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if i % 10 == 0:
                        print ('Step %d: loss_value = %.20f' % (i, loss_value))

                if FLAGS.dependent_student:

                    # self.compute_0filter_of_teacherOutput(sess, feed_dict)

                    self.run_dependent_student(feed_dict, sess, i)

                    if i % 10 == 0:
                        # print("train function: dependent student, multiple optimizers")
                        if FLAGS.num_optimizers == 6:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                            #print ('Step %d: loss_value_softmax = %.20f' % (i, self.loss_value_softmax))
                        else:
                            #print ('Step %d: loss_value_soft = %.20f' % (i, self.loss_value_soft))
                            #print ('Step %d: loss_value_fc3 = %.20f' % (i, self.loss_value_fc3))
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

                    # checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')

                    if FLAGS.teacher:
                        print("save teacher to: "+str(FLAGS.teacher_weights_filename))
                        self.saver.save(sess, FLAGS.teacher_weights_filename)
                    #elif FLAGS.student:
                    #    saver.save(sess, FLAGS.student_filename)
                    #elif FLAGS.dependent_student:
                    #    saver_new = tf.train.Saver()
                    #    saver_new.save(sess, FLAGS.dependent_student_filename)

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

            print(test_accuracy_list)
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
    parser.add_argument('--teacher_weights_filename',type=str,default="./summary-log/teacher_weights_filename_caltech101")
    parser.add_argument('--student_filename',type=str,default="./summary-log/independent_student_weights_filename_caltech101")
    parser.add_argument('--dependent_student_filename',type=str,default="./summary-log/dependent_student_weights_filename_caltech101")
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
    parser.add_argument('--top_1_accuracy',type=bool,help='top-1-accuracy',default=True)
    parser.add_argument('--top_3_accuracy',type=bool,help='top-3-accuracy',default=False)
    parser.add_argument('--top_5_accuracy',type=bool,help='top-5-accuracy',default=False)
    parser.add_argument('--num_iterations',type=int,help='num_iterations',default=1)
    parser.add_argument('--interval_output_train',type=bool,help='interval_output_train',default=False)
    parser.add_argument('--interval_lossValue_train',type=bool,help='interval_lossValue_train',default=False)
    parser.add_argument('--initialization',type=bool,help='initialization',default=False)
    parser.add_argument('--num_optimizers',type=int,help='number of mapping layers from teacher',default=5)

    FLAGS, unparsed = parser.parse_known_args()
    ex = VGG16()
    tf.app.run(main=ex.main, argv=[sys.argv[0]] + unparsed)
