import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def get_mean_and_variance(X, y) :
    data = np.append(X,y,1)
    data_0 = data[data[:, -1] == 0, :-1]
    data_1 = data[data[:, -1] == 1, :-1]

    mu0 = np.mean(data_0, axis =0)
    var0 = np.var(data_0, axis =0)
    mu1 = np.mean(data_1, axis =0)
    var1 = np.var(data_1, axis =0)
    return mu0,var0,mu1,var1

df = pd.read_csv("covid19_metadata.csv")
y = df.iloc[:,-1:]
y = y.replace(to_replace = ['Y','N'],value = [1,0])
X = df.iloc[:,:-1]
X.gender = X.gender.replace(to_replace = ['F','M'],value = [1,0])

[mu0, var0, mu1, var1] = get_mean_and_variance(X, y)

print ("mu0 = ",mu0)
print ("var0 = ",var0)
print ("mu1 = ",mu1)
print ("var1 = ",var1)

#sigma0 = np.sqrt(var0)
#sigma1 = np.sqrt(var1)

#fig, (ax1,ax2) = plt.subplots (1,2, figsize = (12,9))
#xpts0 = np.linspace(mu0[0] - 3*sigma0[0], mu0[0] + 3*sigma0[0], 200)
#xpts1 = np.linspace(mu1[0] - 3*sigma1[0], mu1[0] + 3*sigma1[0], 200)

#ax1.plot(xpts0, norm.pdf(xpts0, mu0[0], sigma0[0]),color = 'black')
#ax1.plot(xpts1, norm.pdf(xpts1, mu1[0], sigma1[0]),color = 'blue')
#ax1.set_title('Gaussian Distribution for Age',fontsize=10)

#ax1.set(xlabel='x', ylabel='Gaussian Distribution')
#ax1.legend(['Not Survived', 'Survived'])

#xpts0 = np.linspace(mu0[1] - 3*sigma0[1], mu0[1] + 3*sigma0[1], 200)
#xpts1 = np.linspace(mu1[1] - 3*sigma1[1], mu1[1] + 3*sigma1[1], 200)
#ax2.plot(xpts0, norm.pdf(xpts0, mu0[1], sigma0[1]),color = 'black')
#ax2.plot(xpts1, norm.pdf(xpts1, mu1[1], sigma1[1]),color = 'blue')
#ax2.set_title('Gaussian Distribution for Gender',fontsize=10)

#ax2.set(xlabel='x', ylabel='Gaussian Distribution')
#ax2.legend(['Not Survived', 'Survived'])
#plt.show()

def learn_reg_params(x, y) :
    y_copy = y[0, 7:]
    y_copy = np.atleast_2d(y_copy).T
    X = np.zeros((x.shape[1] - 7, 14))
    for t in range(7, x.shape[1]):
        X[t - 7,] = np.concatenate([x[0, t - 7:t], y[0, t - 7:t]])
    model = LinearRegression()
    model.fit(X, y_copy)
    y_pred = model.predict(X)
    #print("Learned parameter w = ", model.coef_)
    #print("Learned parameter b = ", model.intercept_)
    #error = y_copy - y_pred
    #me = np.mean(error)
    #var = np.var(error, ddof=1)
    #std = np.sqrt(var)

    #fig, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(12, 9))
    #t = np.arange(8, 221)
    #ax3.scatter(t, y_copy, marker='o', color='blue', linestyle=':')
    #ax3.scatter(t, y_pred, marker='o', color='red', linestyle=':')
    #ax3.set(xlabel='days', ylabel='Death')
    #ax3.legend(['Actual', 'Predicted'])
    #ax3.set_title("Actual and Predicted Death Values")

    #xpts0 = np.linspace(me - 3 * std, me + 3 * std, 200)
    #ax4.plot(xpts0, norm.pdf(xpts0, me, std), color='black')

    #ax4.set_title('Gaussian Distribution for Errors', fontsize=10)
    #ax4.set(xlabel='X', ylabel='Error')

    #xpts0 = np.linspace(me - 3 * std, me + 3 * std, 200)

    #ax5.plot(xpts0, norm.pdf(xpts0, me, std), color='black')
    #ax5.hist(error, bins=70, color='blue', density=True)
    #ax5.set_title('Gaussian Distribution and Histogram plot for Error', fontsize=10)

    #ax5.set(xlabel='X', ylabel='Error')

    #plt.show()
    print("w model coefficient : ", model.coef_)
    print("b model intercept : ", model.intercept_)
    return model.coef_,model.intercept_


df1 = pd.read_csv('covid19_time_series.csv', ',')
data1 = df1.to_numpy()
x = data1[:1,1:]
y = data1[1:,1:]
w,b = learn_reg_params(x, y)

