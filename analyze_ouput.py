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

"""
logName = "output_vgg16/num_of_filter0/log_50per.log"
logging.basicConfig(filename=logName,level=logging.DEBUG)
logging.FileHandler(logName, mode='w')

def loadOutput(filename, findName):
    print(findName)
    logging.info(findName)
    f = open(filename, "r")
    data = f.readlines()
    for i in range(len(data)):
        r = data[i].find(findName)
        if r > -1:
            s = data[i + 1]
            s = json.loads(s)
            break
    f.close()
    print(s)
    print(np.min(s), np.mean(s),stats.mode(s))
    logging.info(str(s))
    logging.info(str(np.min(s))+","+ str(np.mean(s))+","+str(stats.mode(s)))

findNames=['conv1_1', 'conv2_1','conv3_1','conv4_1','conv5_1','conv5_2','conv5_3']
for findName in findNames:
    loadOutput("output_vgg16/num_of_filter0/numOfFliter0_one_batch_50per", findName)
"""
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
        rows = 26
        columns = 10
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
        plt.savefig(sum_pictureName)
        plt.show()

teacher_dirNpy = "output_vgg16/filters_npy/mentor_conv4_1_iteration0.npy"
teacher_dirFigureName = "output_vgg16/images/mentor_conv4_1_iteration0"
#teacher_dirNpy = "output_vgg16/filters_npy/images_feed_iteration0.npy"
#teacher_dirFigureName = "output_vgg16/images/images_feed_iteration0"
#plot_images_afterRelu(teacher_dirNpy, teacher_dirFigureName, flag_norm=True)
plot_images_afterRelu(teacher_dirNpy, teacher_dirFigureName, flag_norm=False)