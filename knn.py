# Author: Suhas Venkatesh Murthy
# Arizona State University

from load_dataset import read
from scipy.stats import mode
from matplotlib import pyplot
import numpy
import operator


def predict(labels):

    y_potential = []
    for i in range(len(labels)):
        y_potential.append(labels[i])

    return int(mode(y_potential)[0][0])


def compute_euclidean_distance(X_train,Y_train,X_test_sample):

    #(a-b)^2 = a^2 + b^2 - 2ab - This is the logic to find -> euclidean distance

    train_data = numpy.matrix(numpy.sum(numpy.square(X_train),axis=1))
    test_data = numpy.matrix(numpy.sum(numpy.square(X_test_sample),axis=1)).T
    dot_component = -2 * numpy.dot(X_train,X_test_sample.T).T
    final_comp = train_data + test_data + dot_component

    all_test_sample_labels = []
    i = 0
    for i in range(final_comp.shape[0]):
        row = final_comp[i].tolist()
        Y_train = list(Y_train)
        temp = zip(row[0],Y_train)
        el_distances = sorted(temp,key=operator.itemgetter(0))
        dists, labels = zip(*el_distances)
        all_test_sample_labels.append((i,labels))
        i+=1
    return all_test_sample_labels


def plot_graph(accuracies):

    k = []
    accuracy = []
    for i in [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]:
        k.append(i)
        accuracy.append(accuracies[i])

    pyplot.plot(k,accuracy)
    pyplot.plot(k, accuracy,'r+')
    pyplot.xlabel('K')
    pyplot.ylabel('Accuracy')
    pyplot.savefig('accuracy_knn.png')
    pyplot.show()


def dum_accuracies_file(accuracies):

    with open('accuracies_knn.txt','w+') as fl:
        for key in accuracies.keys():
            fl.write(str(key))
            fl.write("\t")
            fl.write(str(accuracies[key]) + "\n")


def driver():

    Y_train,X_train = read()
    X_train = X_train.reshape((60000,784))
    X_train = X_train/float(255)

    Y_test,X_test = read("testing")
    X_test = X_test.reshape((10000,784))
    X_test = X_test/float(255)

    Y_pred_k_based = dict()
    accuracies = dict()

    for k in [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]:
        Y_pred_k_based[k] = list()

    labels = compute_euclidean_distance(X_train,Y_train,X_test)

    for i in range(len(X_test)):
        all_l = labels[i][1]
        for k in [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]:
            label = predict(all_l[0:k])
            lst = Y_pred_k_based[k]
            lst.append(label)

    for k in [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]:
        lst_labels = Y_pred_k_based[k]
        count = 0
        for i in range(0,len(X_test)):
            if lst_labels[i] == Y_test[i]:
                count = count+1

        accuracies[k] = count/float(len(X_test))

    plot_graph(accuracies)
    dum_accuracies_file(accuracies)

driver()
