import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,average_precision_score,precision_recall_curve


def logreg_predict_prob(W,b,X) :
    b = np.reshape(b,(len(b),1))
    w = np.transpose(np.concatenate((W,b),axis = 1))
    X_b = np.concatenate((X, np.ones((len(X),1))),axis = 1)
    prob = np.dot(X_b,w)
    len_p = len(prob)
    prob_max = np.zeros((len_p, 1))
    for i in range(0,len_p):
        prob_max[i] = np.max(prob[i,:])
        prob[i,:] = (np.exp(prob[i,:] - prob_max[i]))
        if prob_max[i] < 0 :
            prob[i, :] = (prob[i, :])*(np.exp(prob_max[i])) / (1 + (np.exp(prob_max[i]))*np.sum(prob[i, :]))
        else :
            prob[i,:] = (prob[i,:])/((np.exp(-prob_max[i]))+ np.sum(prob[i,:]))
        #prob[i,:] = prob[i,:]/((np.exp(-prob_max[i]))+np.sum(prob[i,:]))
    k_0 = np.zeros((len_p,1))
    for i in range(len_p):
        k_0[i,:] = 1-np.sum(prob[i,:])
    prob = np.concatenate((prob,k_0),axis = 1)
    return prob


def logreg_fit(X, y, m, eta_start, eta_end, epsilon, max_epoch=1000):

    def loss_sgd(W, b, X, y):
        prob = logreg_predict_prob(W, b, X)
        temp = 0
        for i in range(len(X)):
            result = np.where(prob[i, y[i]] > 0.0000000001, np.log10(prob[i, y[i]]), 0)
            temp = temp + result
        temp = -(temp / len(X))

        ''' 
        for i in range(len(X)):
            temp = temp + np.log(prob[i, y[i]])
        temp = -temp / len(X)
        '''
        return temp

    et = eta_start
    eploss = []
    rows, cols = X.shape
    permute = np.random.permutation(np.arange(0, rows))
    batches = np.array_split(permute, rows / m)
    clas = np.max(y)
    W = np.random.rand(clas, cols)
    b = np.random.rand(clas, 1)

    for epoch in range(max_epoch):
        W_prev = W
        b_prev = b
        for batch in batches:
            Prob = logreg_predict_prob(W, b, X[batch])
            Wb = np.concatenate((W, b), axis=1)
            X_updated = np.concatenate((X, np.ones((rows, 1))), axis=1)
            for i in range(clas):
                Wb[i, :] = Wb[i, :]-et*(np.sum(X_updated[batch]*Prob[:,i,None],axis=0)-np.sum(X_updated[batch][y[batch]== i],axis=0)) / len(batch)

            W = Wb[:, :-1]
            b = Wb[:, -1, None]

        eploss.append(loss_sgd(W, b, X, y))
        if (loss_sgd(W_prev, b_prev, X, y) - loss_sgd(W, b, X, y) < epsilon * loss_sgd(W_prev, b_prev, X, y)):
            et = et / 10
        if (et < eta_end):
            break

    plt.plot(np.arange(len(eploss)), eploss)
    plt.xlabel('Epoch')
    plt.ylabel ('Epoch loss')
    plt.show()

    return W,b


def logreg_predict_class(W, b, X):
    prob = logreg_predict_prob(W, b, X)
    y = np.zeros((len(X),1))
    for i in range(len(X)):
        p_class = 0
        for j in range(len(prob[0])):
            if prob[i, j] >= prob[i, p_class]:
                p_class = j
        y[i] = p_class

    return y

'''
train=pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]

x_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]

y_train = y_train.astype(int)
x_train = x_train.astype(np.float64)
y_test = y_test.astype(int)
x_test = x_test.astype(np.float64)


scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

[G,H] = logreg_fit(x_train,y_train,256,0.01,0.00001,0.0001,1000)
y_p = logreg_predict_class(G,H,x_train)
y_ptest = logreg_predict_class(G,H,x_test)

#print(y_p)

conf_mat = confusion_matrix(y_train, y_p)
conf_mat1 = confusion_matrix(y_test, y_ptest)
print(confusion_matrix(y_test, y_ptest))

acc_conf = np.trace(conf_mat)/np.sum(conf_mat)
acc_conf1 = np.trace(conf_mat1)/np.sum(conf_mat1)
print ()

print (acc_conf)
print (acc_conf1)

print (accuracy_score(y_train, y_p))
print (accuracy_score(y_test, y_ptest))

Prob = logreg_predict_prob(G,H,x_test)
precision, recall, threshold = precision_recall_curve(y_test, Prob[:,1])
avg = average_precision_score(y_test, Prob[:,1])
print ("Average Precision Score is : ", avg)
plt.plot (recall,precision)
plt.xlabel("recall")
plt.ylabel("precision")
plt.show()
'''