from DataPreprocessor import *

from sklearn import svm

from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

C = [0.1, 1, 5, 10, 100]
G = [1/15, 1/10, 1/5, 0.3, 0.5, 1, 2, 5, 10]

fig = plt.figure()

for c in C:
    mae = []
    mse = []
    r2 = []
    for g in G:
        svr = svm.SVR(C=c, gamma=g)
        svr.fit(X_train, Y_train)

        Y_pred_svr = cross_val_predict(svr, X_test, Y_test, cv=10)

        Y_pred_svr = [float(ele) for ele in Y_pred_svr]

        MSE = 0
        MAE = 0
        mean_svr = sum(Y_pred_svr) / len(Y_pred_svr)
        tot = 0
        for i in range(len(Y_test)):
            err = float(Y_test[i]) - Y_pred_svr[i]
            MSE = MSE + err ** 2
            MAE = MAE + abs(err)
            tot = tot + abs(float(Y_test[i]) - mean_svr) ** 2

        mse.append(MSE / len(Y_test))
        mae.append(MAE / len(Y_test))
        r2.append(1 - MSE / tot)
    print(mae)
    plt.plot(G, mae, label=c)

plt.legend(loc='upper right')
plt.xlabel('gamma')
plt.ylabel('MAE')
plt.title('Relation between SVR parameters and MAE')
plt.show()



