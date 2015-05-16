import pandas as pd
import numpy as np
import random
import matplotlib
import time
import cPickle as pickle
from scipy.linalg import hankel
from ggplot import *
from scipy.fftpack import fft

import matplotlib.pyplot as plt

def read_data(Location):
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
    # shift array by key elements to the right
    # e.g shift(1, [1,2,3,4,5]) -> [5,1,2,3,4]
    return array[-key:]+array[:-key]

def save_to_file(data, filename):
    with open(filename, "wb") as output:
        pickle.dump(data, output, 2)

def load_from_file(filename):
    with open(filename, "rb") as output:
        data = pickle.load(output)
    return data

def pause():
    raw_input("PRESS ENTER TO CONTINUE.")

def plot_smoothed_segment(values):
    # plot smoothed segment
    ids = xrange(len(values))
    data = pd.DataFrame({'ids':ids,'values':values})
    plot = ggplot(aes(x='ids', y='values'), data) + \
    stat_smooth(span=0.10) + \
    ggtitle("Smoothed timeseries segment: ")
    print plot

def apply_svd(df, segLength, countSVD, num_max_elems):

    # drop rows with NAs in any column
    df.dropna(inplace=True)

    arr_type = df["Class"]
    arrx = df["x_acc"]
    arry = df["y_acc"]
    arrz = df["z_acc"]

    # find indexes in arrays when type of activity changes:
    idx_x = [0, ]
    idx_x.extend((np.where(arr_type[:-1] != arr_type[1: ])[0] + 1).astype(int).tolist())

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
                print "%s-th segment has been processed..." % k
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

                answ1.append(arr_type[k])

                k = k + segLength
                print "%s-th segment has been processed..." % k
                # pause()

    # print len(x), type(x), len(x[0]), type(x[0])
    # pause()

    filename = "data_%s_%s_%s.pkl" %(segLength, countSVD, num_max_elems)
    save_to_file((x, y, z, answ1), filename)

    return filename

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

def plot_unique_segments(filename):

    x, y, z, answ1 = load_from_file(filename)

    answ1_new = np.array(answ1)
    u, indices = np.unique(answ1_new, return_index=True)
    print indices
    print u

    for i in indices:
        print i
        plot_smoothed_segment(x[i])
        # plot_smoothed_segment(y[i])
        # plot_smoothed_segment(z[i])

def calc_seg_features(ts):
    featVector = []
    featVector.extend((np.mean(ts), # mean
                       np.mean(abs(ts)), # mean of absolute values
                       sum(abs(np.diff(ts))), # sum of absolute difference between consequent elements
                       np.ptp(ts) # difference between max and mean
    ))
    return featVector

def create_features():

    x, y, z, answ1 = load_from_file("data_200_1_3.pkl")
    # def create_features(ts, act_type):

    ts_features = []

    a = x[2]
    # print type(a)
    # with open("data_part.pkl", "wb") as output:
    #     pickle.dump(a, output, 2)

    # for i in xrange(len(x)):
    #     feat_x = calc_seg_features(x[i])
    #     feat_y = calc_seg_features(y[i])
    #     feat_z = calc_seg_features(z[i])
    #
    #     ts_features.append(np.array(feat_x + feat_y + feat_z))
    #     # print ts_features
    #     # pause()
    #
    # print len(ts_features)


    # print a
    # print featVector

    class_list = ["Walking","Jogging","Sitting","Standing","Upstairs","Downstairs"]
    #
    # plt.subplot(2,1,1)
    # plt.plot(a)
    # plt.title(class_list[act_type-1])
    # plt.xlabel('Series')
    #
    # plt.subplot(2,1,2)
    # plt.plot(freq, np.abs(signal))
    # plt.xlabel('Frequency')
    # plt.show()
    #
    # max_freq_idx = np.abs(signal).argsort()[-2:]
    # print 1/freq[max_freq_idx]

    # with open("data_features.pkl", "wb") as output:
    #     pickle.dump((x, y, z, answ1), output, 2)

# def print_series():

def main():
    Location = './my_data.csv'
    data = read_data(Location)
    # data, labels = rep_sep_classes(data)
    # filename = apply_svd(data, 200, 1, 3)
    plot_unique_segments("data_200_1_3.pkl")
    # create_features()

main()