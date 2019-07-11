import numpy as np
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import linalg as LA
from os import listdir
import logging
import json
from scipy import stats
import os

def l2_norm_one_batch(data):
    print("original data.shape: " + str(data.shape))
    data = data.transpose(0, 3, 1, 2)
    norm = LA.norm(data, axis=(0))
    data_norm = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    for i in range(data.shape[0]):
        data_norm[i] = np.true_divide(data[i], norm)
        #data_norm[i] = data[i] / norm
        data_norm[i][~np.isfinite(data_norm[i])] = 0.
    data_norm_new = data_norm.transpose(0, 2, 3, 1)
    print("data.shape: "+ str(data.shape))
    #print(data)
    print("norm.shape: " + str(norm.shape))
    #print(norm)
    print("data_norm.shape: " + str(data_norm.shape))
    #print(data_norm)
    print("data_norm_new.shape: " + str(data_norm_new.shape))
    return data_norm_new

def plot_images_afterRelu(dirNpy, dirName, flag_norm=False):
    output = np.load(dirNpy)
    print(output.shape)
    if flag_norm:
        output = l2_norm_one_batch(output)
    #print(output_wrn.shape)

    # for one image
    for i in range(1):
        print("iteration: " + str(i))
        print("The shape of one image: " + str(output[i].shape))

        img = output[i]
        img = img.transpose(2, 0, 1)
        img_sum = np.sum(img, axis=0)
        fig = plt.figure(figsize=(80, 80))
        rows = 10
        columns = 7
        for j in range(img.shape[0]):
        #for j in range(1):
            img_dim = img[j]
            #img_dim = np.square(np.abs(img_dim))
            #np.savetxt("output_wrn/csv/img_dim"+str(j)+".csv", img_dim, delimiter=",")
            #print(img_dim)
            #fig.add_subplot(rows, columns, j+1).title.set_text("filter"+str(j))
            fig.add_subplot(rows, columns, j + 1)
            plt.imshow(img_dim)

        fig.add_subplot(rows, columns, img.shape[0]+1).title.set_text("sum")
        plt.imshow(img_sum)
        plt.tight_layout()
        #pictureName = dirName + str(i) + "_norm" + str(flag_norm) + ".png"
        #plt.savefig(pictureName)
        plt.show()

        plt.imshow(img_sum)
        sum_pictureName = dirName + str(i) + "_norm" + str(flag_norm) + "_sum" + ".png"
        #plt.savefig(sum_pictureName)
        plt.show()

def count_filter0_num_convLayers(output, perNum):
    filter_count = []
    for i in range(output.shape[0]):
        img = output[i]
        img = img.transpose(2, 0, 1)
        count = 0
        for j in range(img.shape[0]):

            # count number filters whose 90% output_wrn are 0
            num_sum = float(img[j].shape[0] * img[j].shape[1])
            count0_perFIlter = (num_sum - np.count_nonzero(img[j])) / num_sum
            if count0_perFIlter >= perNum:
                count = count + 1

        filter_count.append(count)
    return filter_count

def count_filter0_num_fcLayers(output, about0num):
    print(output.shape)
    filter_count = []
    for i in range(output.shape[0]):
        #print(output[i].shape
        count0Num_perImg = 0
        for j in range(output.shape[1]):
            if output[i][j] <= about0num:
                count0Num_perImg = count0Num_perImg + 1
        filter_count.append(count0Num_perImg)
        #print(np.max(output[i]), np.min(output[i]))
    return filter_count

def loadOutputLog_calcuteMean(filename, searchName):
    logging.info(searchName)
    f = open(filename, "r")
    data = f.readlines()
    for i in range(len(data)):
        r = data[i].find(searchName)
        if r > -1:
            s = data[i + 1].split(":")[2]
            s = json.loads(s)
            break
    f.close()
    logging.info(s)
    logging.info("Mean: {}, Min: {}, Mode: {}.\n".format(np.mean(s), np.min(s), stats.mode(s)))

def readNpyFile_count0Filters_saveToLog_calculateMean(logName, readNpyPath, perOf0Filters):
    logging.basicConfig(filename=logName, level=logging.DEBUG)
    logging.FileHandler(logName, mode='w')

    searchNames = []
    for filename in sorted(os.listdir(readNpyPath)):
        logging.info(filename)
        output = np.load(readNpyPath + filename)
        if "conv" in readNpyPath:
            filter_count = count_filter0_num_convLayers(output, perOf0Filters)
        elif "fc" in readNpyPath:
            filter_count = count_filter0_num_fcLayers(output, perOf0Filters)
        logging.info(filter_count)
        searchNames.append(filename)

    #searchNames = ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"]
    logging.info("\n\nBegin to compute mean, min, and mode of every layer's output")
    for searchName in searchNames:
        loadOutputLog_calcuteMean(logName, searchName)

teacher_dirNpy = "output_vgg16/filters_npy/mentor_conv1_1_iteration0.npy"
teacher_dirFigureName = "output_vgg16/images/mentor_conv1_1_iteration0"
#teacher_dirNpy = "output_vgg16/filters_npy/images_feed_iteration0.npy"
#teacher_dirFigureName = "output_vgg16/images/images_feed_iteration0"
#plot_images_afterRelu(teacher_dirNpy, teacher_dirFigureName, flag_norm=True)
#plot_images_afterRelu(teacher_dirNpy, teacher_dirFigureName, flag_norm=False)

readNpyPath = "output_vgg16/filters_npy_conv/"
logName = "output_vgg16/num_of_filter0/conv/log_0.5.log"
perOf0Filters = 0.5
readNpyFile_count0Filters_saveToLog_calculateMean(logName, readNpyPath, perOf0Filters)

readNpyPath_fc = "output_vgg16/filters_npy_fc/"
logName_fc = "output_vgg16/num_of_filter0/fc/log_1.0.log"
about0num = 1.0
#readNpyFile_count0Filters_saveToLog_calculateMean(logName_fc, readNpyPath_fc, about0num)
