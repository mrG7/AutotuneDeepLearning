import pandas as pd
import numpy as np
import random
import matplotlib
import time
import cPickle as pickle
from scipy.linalg import hankel
from ggplot import *

import matplotlib.pyplot as plt

def read_data(Location):
    # f1 = open(Location, 'r')
    # f2 = open('yourBigFile.csv', 'w')
    # for line in f1:
    #     f2.write(line.replace(';', ''))
    # f1.close()
    # f2.close()

    f = open(Location, 'r')
    columns=["User","Class","Timestamp","x_acc","y_acc","z_acc"]
    df = pd.read_csv(Location, names = columns, engine = 'python', sep=',|;')
    f.close()
    print "Data is read..."

    return df

def rep_sep_classes(df):
    # replacing classes with numbers
    class_list = ["Walking","Jogging","Sitting","Standing","Upstairs","Downstairs"]
    labels_list = [1, 2, 3, 4, 5, 6]
    df = df.replace(class_list, labels_list)
    labels = np.array(df["Class"])
    # print df.head()
    # df = df.drop(["Class"], 1)
    print "Classes is replaced with numbers and separated..."

    return df, labels

def shift(key, array):
    return array[-key:]+array[:-key]

# def plot_segment(values, title_y, activity_class):
def plot_segment(values):

    ids = xrange(len(values))
    data = pd.DataFrame({'ids':ids,'values':values})
    plot = ggplot(aes(x='ids', y='values'), data) + \
    stat_smooth(span=0.10) + \
    ggtitle("Smoothed timeseries segment: ")
    print plot

def apply_svd(df, segLength, countSVD, num_max_elems):

    df.dropna(inplace=True)

    arr_type = df["Class"]
    arrx = df["x_acc"]
    arry = df["y_acc"]
    arrz = df["z_acc"]

    # inds = pd.isnull(df).any(1).nonzero()[0]
    # print inds

    # print arrz
    # arrz = arrz.map(lambda x: x.rstrip(';'))
    # print arrz[343396:343500]
    # raw_input("PRESS ENTER TO CONTINUE.")

    # find indexes in arrays when type of activity changes:
    idx_x = [0, ]
    idx_x.extend((np.where(arr_type[:-1] != arr_type[1: ])[0] + 1).astype(int).tolist())

    # idx_x = idx_x[-5:]
    # print idx_x

    print "There are %s moments in the dataset when the human activity changes: " % len(idx_x)
    print idx_x
    print "Number of elements in the dataset: %s" %len(arrx)
    arrx = arrx.tolist()
    arry = arry.tolist()
    arrz = arrz.tolist()


    x = []
    y = []
    z = []
    answ1 = []

    for i in range(len(idx_x)): # cycle through moments of activity changing
        k = idx_x[i] # current moment (id in the dataset)

        # if the current moment is the last one
        if i == len(idx_x)-1:
            limit = len(arrx)-1 # pick the last index in the dataset as the segmentation limit
        else:
            limit = idx_x[i+1] # pick next moment of the activity change

        while k + segLength < limit:
            if countSVD == 0:
                x.append(arrx[k:(k + segLength - 1)])
                y.append(arry[k:(k + segLength - 1)])
                z.append(arrz[k:(k + segLength - 1)])
                answ1.append(arr_type[k])
                k = k + segLength
            else:
                testx = arrx[k:(k + segLength - 1)]
                testy = arry[k:(k + segLength - 1)]
                testz = arrz[k:(k + segLength - 1)]
                # print "Timeseries segment:\n%s\n" % test

                matx = hankel(testx, shift(1,testx))
                maty = hankel(testy, shift(1,testy))
                matz = hankel(testz, shift(1,testz))
                # print "Hankel matrix: \n%s\n" % matx

                x1 = get_svd(matx, num_max_elems)
                y1 = get_svd(maty, num_max_elems)
                z1 = get_svd(matz, num_max_elems)

                x.append(x1)
                y.append(y1)
                z.append(z1)
                # print "X: \n%s\n" % x

                # y = [y, y1']
                # z = [z, z1']
                answ1.append(arr_type[k])
                # print answ1

                k = k + segLength
                print k
                # raw_input("PRESS ENTER TO CONTINUE.")

    output = open('data.pkl', 'wb')
    pickle.dump((x, y, z, answ1), output, 2)
    output.close()

    # with open("data.pkl", "wb") as output:
    #     pickle.dump((x, y, z, answ1), output, 2)
    #     output.close()

    # print len(x), len(answ1)
    # print x[0], len(x[0])
    # print answ1
    # plot_segment(x[38])

    # answ1_new = np.array(answ1)
    # u, indices = np.unique(answ1_new, return_index=True)
    # print indices
    #
    # print u
    # class_list = ["Walking","Jogging","Sitting","Standing","Upstairs","Downstairs"]
    # labels_list = [1, 2, 3, 4, 5, 6]

    # for i in indices:
    #     plot_segment(x[i])

def get_svd(a, num_max_elems):

    U, s, V = np.linalg.svd(a)
    assert np.allclose(a, np.dot(U, np.dot(np.diag(s), V)))
    # print U.shape, V.shape, s.shape

    # print "Singular values:\n%s\n" % s
    s[num_max_elems:] = 0
    # print "Singular values (%s max elements):\n%s\n" %(num_max_elems, s)
    mat = np.dot(U, np.dot(np.diag(s), V))
    # print "Recovered matrix:\n%s" % mat

    x = mat[0, :]

    return x

def plot_unique_segments():
    with open("data.pkl", "rb") as output:
        x, y, z, answ1 = pickle.load(output)
        output.close()

    answ1_new = np.array(answ1)
    u, indices = np.unique(answ1_new, return_index=True)
    print indices
    print u

    # class_list = ["Walking","Jogging","Sitting","Standing","Upstairs","Downstairs"]

    # print answ1

    for i in indices:
        # if i != 0:
        #     ind = i-1
        # else:
        #     ind = i
        # answ1[ind]]
        print i
        # plot_segment(x[i], 'x_acc', class_list[answ1[i]])
        # plot_segment(y[i], 'y_acc', class_list[answ1[i]])
        # plot_segment(z[i], 'z_acc', class_list[answ1[i]])
        plot_segment(x[i])
        plot_segment(y[i])
        plot_segment(z[i])

Location = './my_data.csv'
data = read_data(Location)
data, labels = rep_sep_classes(data)
apply_svd(data, 200, 0, 3)
plot_unique_segments()

# with open("data.pkl", "rb") as output:
#     x, answ1 = pickle.load(output)
#     output.close()

# answ1_new = np.array(answ1)
# u, indices = np.unique(answ1_new, return_index=True)
# print indices
#
# for i in indices:
#     plot_segment(x[i])