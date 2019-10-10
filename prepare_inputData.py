import os
import sys
"""
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
"""
def create_cifar10_txtfile_allData(openFile, writeFile):
    file_writer = open(writeFile, "w")
    for subdir, dirs, files in os.walk(openFile):
        print(subdir, dirs, files)
        for filename in files:
            num, name = filename.split("_")
            labelname, png = str(name).split(".")
            print(num)
            print(labelname)
            print(png)

            if labelname == "airplane":
                writeName = str(0) + "," + str(subdir) + "/" + str(filename) + "\n"
            elif labelname == "automobile":
                writeName = str(1) + "," + str(subdir) + "/" + str(filename) + "\n"
            elif labelname == "bird":
                writeName = str(2) + "," + str(subdir) + "/" + str(filename) + "\n"
            elif labelname == "cat":
                writeName = str(3) + "," + str(subdir) + "/" + str(filename) + "\n"
            elif labelname == "deer":
                writeName = str(4) + "," + str(subdir) + "/" + str(filename) + "\n"
            elif labelname == "dog":
                writeName = str(5) + "," + str(subdir) + "/" + str(filename) + "\n"
            elif labelname == "frog":
                writeName = str(6) + "," + str(subdir) + "/" + str(filename) + "\n"
            elif labelname == "horse":
                writeName = str(7) + "," + str(subdir) + "/" + str(filename) + "\n"
            elif labelname == "ship":
                writeName = str(8) + "," + str(subdir) + "/" + str(filename) + "\n"
            else:
                writeName = str(9) + "," + str(subdir) + "/" + str(filename) + "\n"

            file_writer.write(writeName)
            print("\n")


def select_subset(readFileName, writeFileName, target_labels):
    f = open(readFileName)
    lines = f.readlines()
    sublines = []
    for line in lines:
        if int(line.split(",")[0]) in target_labels:
            sublines.append(line)
    f.close()

    f = open(writeFileName, "w")
    f.writelines(sublines)
    f.close()

    print("The number of data from readFile: "+str(len(lines)))
    print("The number of data from writeFile: "+str(len(sublines)))


#create_cifar10_txtfile_allData("./cifar10_images/test", "./cifar10_input/cifar10-test.txt")
#create_cifar10_txtfile_allData("./cifar10_images/train", "./cifar10_input/cifar10-train.txt")

#select_subset("./cifar10_input/cifar10-train.txt", "./cifar10_input/cifar10-train-7categories06.txt",target_labels=[0,1,2,3,4,5,6])
#select_subset("./cifar10_input/cifar10-test.txt", "./cifar10_input/cifar10-test-3categories789.txt", target_labels=[7,8,9])

select_subset("./cifar10_input/cifar10-train.txt", "./cifar10_input/cifar10-train-3categories789.txt",target_labels=[7,8,9])
