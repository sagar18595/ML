import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import accuracy_score


def knn_classifier(X_train, y_train, X_test, k):
    def eucl_dist(A, B):
        dist = np.sqrt(np.sum((A - B) ** 2))
        return dist

    final_label = []
    idxs = []
    for pt in X_test:
        pt_d = []
        for i in range(len(X_train)):
            distances = eucl_dist(np.array(X_train[i, :]), pt)
            pt_d.append(distances)
        pt_d = np.array(pt_d)
        Idxs = np.arange(0, len(X_train))
        dist = np.argsort(pt_d)[:k]

        neighbors = Idxs[dist].T
        labels = y_train[dist]
        idxs.append(neighbors)

        vote = mode(labels)
        vote = vote.mode[0]
        final_label.append(vote)

    return final_label, idxs

train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

X_train = np.array(train.iloc[:,1:].values).astype(int)
y_train = np.array(train['label'].values).astype(int)
X_test = np.array(test.iloc[:,1:].values).astype(int)
y_test = np.array(test['label'].values).astype(int)

X_train = (X_train/255).astype(np.float64)
X_test = (X_test/255).astype(np.float64)

y_pred, idxs = knn_classifier(X_train,y_train, X_test,5)
print ("Accuracy", accuracy_score(y_test,y_pred))

m = [0, 1, 2, 4, 8, 16, 32]
k_list = [x * 2 + 1 for x in m]
acc_list = []
for i in k_list :
    y_pred, idxs = knn_classifier(X_train,y_train,X_test, i)
    a = accuracy_score(y_test,y_pred)
    acc_list.append(a)

plt.plot (k_list,acc_list, marker='o', color='blue', mfc='red')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

n = [100,200,400,600,800,1000]
acc = []
for i in n :
    y_pred, idxs = knn_classifier(X_train[:i,:], y_train[:i], X_test, 3)
    a = accuracy_score(y_test,y_pred)
    acc.append(a)

plt.plot (n,acc, marker='o', color='b', mfc='black')
plt.xlabel("N")
plt.ylabel("Accuracy")
plt.show()

def knn_classifier1(X_train, y_train, X_test, k):
    def m_dist(A, B):
        dist = np.sum(abs(A - B))
        return dist

    final_label = []
    idxs = []
    for pt in X_test:
        pt_d = []
        for i in range(len(X_train)):
            distances = m_dist(np.array(X_train[i, :]), pt)
            pt_d.append(distances)
        pt_d = np.array(pt_d)
        Idxs = np.arange(0, len(X_train))
        dist = np.argsort(pt_d)[:k]

        neighbors = Idxs[dist].T
        labels = y_train[dist]
        idxs.append(neighbors)

        vote = mode(labels)
        vote = vote.mode[0]
        final_label.append(vote)

    return final_label, idxs

y_pred1,Idxs1 = knn_classifier1(X_train,y_train, X_test,3)
y_pred, Idxs = knn_classifier(X_train,y_train, X_test,3)
print ("Accuracy score for Manhattan distance : ", accuracy_score(y_test,y_pred1))
print ("Accuracy score for Eucledean distance : ", accuracy_score(y_test,y_pred))

''' 
y_pred1, Idxs1 = knn_classifier(X_train,y_train, X_test,5)
acc = []
for i in range (len(y_test)) :
    if (y_pred1[i] != y_test[i]) :
        acc.append(i)
        if len(acc) > 2 :
            break

fig = plt.figure()
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.imshow(np.reshape(X_test[i], (28, 28)))
    print("neighbour id for Sample " + str(i + 1) + " is " + str(Idxs1[i]))
    print()
    plt.title("Incorrectly Predicted Label/Value in Xtest for Sample " + str(i + 1) + " is " + str(y_pred1[acc[i]]))
    plt.axis('off')

#plt.tight_layout()
plt.show()

for i in acc :
    fig = plt.figure()
    print ("neighbors for Sample " + str(acc.index(i) + 1))
    for j in range (5) :
        m = Idxs1[i][j]
        plt.subplot(3,5,j+1)
        plt.imshow(np.reshape(X_train[m], (28,28)))
        plt.title ("neighbor " + str(j+1))
        plt.axis('off')
#plt.tight_layout()
plt.show()

'''