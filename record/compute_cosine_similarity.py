import tensorflow as tf

def find_largest_cosine_among_two_numbers(num1, num2):

    def return_1():
        return tf.constant(1)

    def return_2():
        return tf.constant(2)

    result = tf.cond(num1 > num2, return_1, return_2)

    return result

def find_largest_cosine_among_three_numbers(num1, num2, num3):

    def return_1():
        return tf.constant(1)

    def return_2():
        return tf.constant(2)

    def return_3():
        return tf.constant(3)

    def compare_number13(num1, num3):
        result = tf.cond(num1 > num3, return_1, return_3)
        return result

    def compare_two_number23(num2, num3):
        result = tf.cond(num2 > num3, return_2, return_3)
        return result

    result = tf.cond(num1>num2, lambda: compare_number13(num1, num3), lambda: compare_two_number23(num2, num3))

    return result

def cosine_similarity_of_same_width(mentee_data_dict, mentor_data_dict, sess, feed_dict, num_optimizers):
    """
        cosine similarity is calculated between 1st layer of mentee and 1st layer of mentor.
        Similarly, cosine similarity is calculated between 1st layer of mentee and 2nd layer of mentor.
    """
    normalize_a_1 = tf.nn.l2_normalize(mentee_data_dict.conv1_1, 0)
    normalize_b_11 = tf.nn.l2_normalize(mentor_data_dict.conv1_1, 0)
    normalize_b_12 = tf.nn.l2_normalize(mentor_data_dict.conv1_2, 0)

    normalize_a_2 = tf.nn.l2_normalize(mentee_data_dict.conv2_1, 0)
    normalize_b_21 = tf.nn.l2_normalize(mentor_data_dict.conv2_1, 0)
    normalize_b_22 = tf.nn.l2_normalize(mentor_data_dict.conv2_2, 0)

    normalize_a_3 = tf.nn.l2_normalize(mentee_data_dict.conv3_1, 0)
    normalize_b_31 = tf.nn.l2_normalize(mentor_data_dict.conv3_1, 0)
    normalize_b_32 = tf.nn.l2_normalize(mentor_data_dict.conv3_2, 0)
    normalize_b_33 = tf.nn.l2_normalize(mentor_data_dict.conv3_3, 0)

    cosine1_11 = tf.reduce_sum(tf.multiply(normalize_a_1, normalize_b_11))
    cosine1_12 = tf.reduce_sum(tf.multiply(normalize_a_1, normalize_b_12))
    cosine1 = find_largest_cosine_among_two_numbers(cosine1_11, cosine1_12)

    cosine2_21 = tf.reduce_sum(tf.multiply(normalize_a_2, normalize_b_21))
    cosine2_22 = tf.reduce_sum(tf.multiply(normalize_a_2, normalize_b_22))
    cosine2 = find_largest_cosine_among_two_numbers(cosine2_21, cosine2_22)

    cosine3_31 = tf.reduce_sum(tf.multiply(normalize_a_3, normalize_b_31))
    cosine3_32 = tf.reduce_sum(tf.multiply(normalize_a_3, normalize_b_32))
    cosine3_33 = tf.reduce_sum(tf.multiply(normalize_a_3, normalize_b_33))
    cosine3 = find_largest_cosine_among_three_numbers(cosine3_31, cosine3_32, cosine3_33)

    if num_optimizers==5:
        normalize_a_4 = tf.nn.l2_normalize(mentee_data_dict.conv4_1, 0)
        normalize_b_41 = tf.nn.l2_normalize(mentor_data_dict.conv4_1, 0)
        normalize_b_42 = tf.nn.l2_normalize(mentor_data_dict.conv4_2, 0)
        normalize_b_43 = tf.nn.l2_normalize(mentor_data_dict.conv4_3, 0)

        normalize_a_5 = tf.nn.l2_normalize(mentee_data_dict.conv5_1, 0)
        normalize_b_51 = tf.nn.l2_normalize(mentor_data_dict.conv5_1, 0)
        normalize_b_52 = tf.nn.l2_normalize(mentor_data_dict.conv5_2, 0)
        normalize_b_53 = tf.nn.l2_normalize(mentor_data_dict.conv5_3, 0)

        cosine4_41 = tf.reduce_sum(tf.multiply(normalize_a_4, normalize_b_41))
        cosine4_42 = tf.reduce_sum(tf.multiply(normalize_a_4, normalize_b_42))
        cosine4_43 = tf.reduce_sum(tf.multiply(normalize_a_4, normalize_b_43))
        cosine4 = find_largest_cosine_among_three_numbers(cosine4_41, cosine4_42, cosine4_43)

        cosine5_51 = tf.reduce_sum(tf.multiply(normalize_a_5, normalize_b_51))
        cosine5_52 = tf.reduce_sum(tf.multiply(normalize_a_5, normalize_b_52))
        cosine5_53 = tf.reduce_sum(tf.multiply(normalize_a_5, normalize_b_53))
        cosine5 = find_largest_cosine_among_three_numbers(cosine5_51, cosine5_52, cosine5_53)

    if num_optimizers == 3:
        cosine = [cosine1, cosine2, cosine3]
    if num_optimizers == 5:
        cosine = [cosine1, cosine2, cosine3, cosine4, cosine5]

    """
    print("start")

    print("1th")
    print sess.run([cosine1_11,cosine1_12, cosine1], feed_dict=feed_dict)

    print("2th")
    print sess.run([cosine2_21,cosine2_22, cosine2], feed_dict=feed_dict)

    print("3th")
    print sess.run([cosine3_31,cosine3_32, cosine3_33, cosine3], feed_dict=feed_dict)


    print("4th")
    print sess.run([cosine4_41,cosine4_42, cosine4_43, cosine4], feed_dict=feed_dict)

    print("5th")
    print sess.run([cosine5_51,cosine5_52, cosine5_53, cosine5], feed_dict=feed_dict)

    print("ended")
    """

    return cosine

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


