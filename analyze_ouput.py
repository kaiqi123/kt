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