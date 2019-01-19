from DataPreprocessor import *
# Importing Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

svr = svm.SVR(C=5, gamma=5)
svr.fit(X_train, Y_train)

knn = KNeighborsClassifier(n_neighbors=53, weights='distance')
knn.fit(X_train, Y_train)

rf = RandomForestClassifier(n_estimators=115, criterion='entropy', random_state=0,
                            class_weight='balanced',
                            max_features='auto')
rf.fit(X_train, Y_train)


Y_pred_svr = svr.predict(X_test)
Y_pred_knn = knn.predict(X_test)
Y_pred_rf = rf.predict(X_test)

Y_test = [float(ele) for ele in Y_test]
Y_pred_svr = [float(ele) for ele in Y_pred_svr]
Y_pred_knn = [float(ele) for ele in Y_pred_knn]
Y_pred_rf = [float(ele) for ele in Y_pred_rf]


fig = plt.figure()
t = np.arange(2, 11, 1)
plt.plot(t, t, 'k--')
plt.plot(np.array(Y_test), np.array(Y_pred_svr), 'b.', label="SVR")
plt.plot(np.array(Y_test), np.array(Y_pred_knn), 'c.', label="KNN")
plt.plot(np.array(Y_test), np.array(Y_pred_rf), 'y.', label="RF")
plt.axis([2, 10, 2, 10])
plt.xlabel('measured')
plt.ylabel('predicted')
plt.legend(loc='upper left')
plt.show()

final_accuracy = 0
MSE = 0
MAE = 0
Y_error_svr = {}
mean_svr = sum(Y_pred_svr) / len(Y_pred_svr)
tot = 0
for i in range(len(Y_test)):
    err = Y_test[i] - Y_pred_svr[i]
    MSE = MSE + err**2
    MAE = MAE + abs(err)
    tot = tot + abs(Y_test[i] - mean_svr) ** 2
    if Y_test[i] not in Y_error_svr.keys():
        Y_error_svr[Y_test[i]] = []
    Y_error_svr.get(Y_test[i]).append(abs(err))

    if abs(err) <= 1:
        final_accuracy += 1

for key in Y_error_svr.keys():
    Y_error_svr[key] = sum(Y_error_svr[key]) / len(Y_error_svr[key])

print('SVR Accuracy = ', final_accuracy / len(Y_test) * 100, '%')
print('SVR RMSE = ', (MSE / len(Y_test)) ** 0.5)
print('SVR MAE = ', MAE / len(Y_test))
print('SVR R2 = ', 1 - MSE / tot)
score1 = sorted(list(Y_error_svr.keys()))
error_svr = [Y_error_svr.get(ele) for ele in score1]


final_accuracy = 0
MSE = 0
MAE = 0
Y_error_knn = {}
mean_knn = sum(Y_pred_knn) / len(Y_pred_knn)
tot = 0
for i in range(len(Y_test)):
    err = Y_test[i] - Y_pred_knn[i]
    MSE = MSE + err ** 2
    MAE = MAE + abs(err)
    tot = tot + abs(Y_test[i] - mean_knn) ** 2
    if Y_test[i] not in Y_error_knn.keys():
        Y_error_knn[Y_test[i]] = []
    Y_error_knn.get(Y_test[i]).append(abs(err))

    if abs(err) <= 1:
        final_accuracy += 1

for key in Y_error_knn.keys():
    Y_error_knn[key] = sum(Y_error_knn[key]) / len(Y_error_knn[key])

print('KNN Accuracy = ', final_accuracy / len(Y_test) * 100, '%')
print('KNN RMSE = ',  (MSE / len(Y_test)) ** 0.5)
print('KNN MAE = ', MAE / len(Y_test))
print('KNN R2 = ', 1 - MSE / tot)
score2 = sorted(list(Y_error_knn.keys()))
error_knn = [Y_error_knn.get(ele) for ele in score2]

final_accuracy = 0
MSE = 0
MAE = 0
Y_error_rf = {}
mean_rf = sum(Y_pred_rf) / len(Y_pred_rf)
tot = 0
for i in range(len(Y_test)):
    err = Y_test[i] - Y_pred_rf[i]
    MSE = MSE + err ** 2
    MAE = MAE + abs(err)
    tot = tot + abs(Y_test[i] - mean_rf) ** 2
    if Y_test[i] not in Y_error_rf.keys():
        Y_error_rf[Y_test[i]] = []
    Y_error_rf.get(Y_test[i]).append(abs(err))

    if abs(err) <= 1:
        final_accuracy += 1

for key in Y_error_rf.keys():
    Y_error_rf[key] = sum(Y_error_rf[key]) / len(Y_error_rf[key])

print('RF Accuracy = ', final_accuracy / len(Y_test) * 100, '%')
print('RF RMSE = ', (MSE / len(Y_test)) ** 0.5)
print('RF MAE = ', MAE / len(Y_test))
print('RF R2 = ', 1 - MSE / tot)
score3 = sorted(list(Y_error_rf.keys()))
error_rf = [Y_error_rf.get(ele) for ele in score3]

fig = plt.figure()
plt.plot(np.array(score1), np.array(error_svr), 'b', label='SVR')
plt.plot(np.array(score2), np.array(error_knn), 'c', label='KNN')
plt.plot(np.array(score3), np.array(error_rf), 'y', label='RF')
plt.xlabel('Score')
plt.ylabel('Bias')
plt.legend(loc='upper right')
plt.show()

fig = plt.figure()
plt.plot(np.array(score1), np.array(error_svr), 'b', label='SVR')
plt.plot(np.array(score2), np.array(error_knn), 'c', label='KNN')
plt.plot(np.array(score3), np.array(error_rf), 'y', label='RF')
plt.axis([5, 9, 0, 1.5])
plt.xlabel('Score')
plt.ylabel('Bias')
plt.legend(loc='upper right')
plt.show()

acc_svr = []
acc_knn = []
acc_rf = []

rg = np.arange(0.2, 1.1, 0.05)
for thre in rg:
    count = 0
    for i in range(len(Y_test)):
        e = abs(Y_test[i] - Y_pred_svr[i])
        if e <= thre:
            count = count + 1
    acc_svr.append(count / len(Y_test) * 100)

    count = 0
    for i in range(len(Y_test)):
        e = abs(Y_test[i] - Y_pred_knn[i])
        if e <= thre:
            count = count + 1
    acc_knn.append(count / len(Y_test) * 100)

    count = 0
    for i in range(len(Y_test)):
        e = abs(Y_test[i] - Y_pred_rf[i])
        if e <= thre:
            count = count + 1
    acc_rf.append(count / len(Y_test) * 100)

fig = plt.figure()
plt.plot(rg, np.array(acc_svr), 'b', label='SVR')
plt.plot(rg, np.array(acc_knn), 'c', label='KNN')
plt.plot(rg, np.array(acc_rf), 'y', label='RF')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()



