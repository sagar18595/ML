import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import cdist


def k_means(x, k, no_of_iterations=30, error = False):
    idx = np.random.choice(len(x), k, replace=False)
    # Randomly choosing Centroids
    centroids = x[idx, :]  # Step 1

    # finding the distance between centroids and all the data points
    distances = cdist(x, centroids, 'euclidean')  # Step 2

    # Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances])  # Step 3

    sse = []
    it = []
    # Repeating the above steps for a defined number of iterations
    # Step 4
    for iter1 in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp = x[points == idx].mean(axis=0)
            centroids.append(temp)

        centroids = np.vstack(centroids)  # Updated Centroids

        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        sse_1 = np.array([np.min(i) for i in distances])
        #print("Iter, SSE -> ", (iter1, np.sum(sse_1 ** 2)))
        it.append(iter1)
        sse.append(np.sum(sse_1 ** 2))

    if error == True :
        plt.plot(it, sse)
        plt.xlabel("Iter")
        plt.ylabel("SSE")
        plt.show()

    return centroids

def assignment(X, centroids):
    distances = cdist(X, centroids ,'euclidean')
    points = np.array([np.argmin(i) for i in distances])
    return points

if __name__ == '__main__':
    '''
    train = np.genfromtxt('mnist_train_hw5.csv', delimiter=',', skip_header=1)
    test = np.genfromtxt('mnist_test_hw5.csv', delimiter=',', skip_header=1)

    Xtrain = train[:, 1:]
    ytrain = train[:, 0]
    Xtest = test[:, 1:]
    ytest = test[:, 0]

    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    centroid = k_means(Xtrain, 10,error= True)
    ind = assignment(Xtest, centroid)

    acc = Counter(ind)
    print("Counter : ", acc)

    fig = plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.reshape(centroid[i], (28, 28)))
        plt.title(acc[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    centroid = k_means(Xtrain, 8,error= True)
    ind = assignment(Xtest, centroid)

    acc = Counter(ind)
    print("Counter : ", acc)

    fig = plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(np.reshape(centroid[i], (28, 28)))
        plt.title(acc[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    centroid = k_means(Xtrain, 12, error= True)
    ind = assignment(Xtest, centroid)

    acc = Counter(ind)
    print("Counter : ", acc)

    fig = plt.figure()
    for i in range(12):
        plt.subplot(2, 6, i + 1)
        plt.imshow(np.reshape(centroid[i], (28, 28)))
        plt.title(acc[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    '''
