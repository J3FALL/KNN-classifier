import numpy as np
import math
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from mpl_toolkits.mplot3d import axes3d, Axes3D


def colorData(data):
    colors = ["green", "red"]
    color_point = []
    for point in data:
        color_point.append(colors[int(point[2])])
    return color_point


def toPolar(data):
    result = []
    for point in data:
        result.append([math.atan(abs(point[1] / point[0])), math.sqrt(point[0] ** 2 + point[1] ** 2)])
    result = np.array(result)
    return result


def centerShift(data):
    center = []
    data_ = np.array(deepcopy(data))
    center.append(sum(data_[:, 0]) / len(data_[:, 0]))
    center.append(sum(data_[:, 1]) / len(data_[:, 1]))
    for row in data_:
        row[0] = row[0] - center[0]
        row[1] = row[1] - center[1]
    return data_


def dist(point1, point2, m):
    ans = math.pow(sum([abs(x - y) ** m for x, y in zip(point1, point2)]), (1 / m))
    return ans


def normilize(X):
    X_res = deepcopy(X)
    X_res = np.array(X_res)
    xmax = X_res.max(axis=0)
    it = 0
    for param in xmax:
        X_res[:, it] = X[:, it] / param
        it += 1
    return X_res


def splitTrainTest(kNum, K, X, y):
    start = kNum * K
    end = (kNum + 1) * K
    trainDataX = np.concatenate((X[0:start], X[end:]))
    testDataX = X[start:end]
    trainDataY = np.concatenate((y[0:start], y[end:]))
    testDataY = y[start:end]
    return trainDataX, testDataX, trainDataY, testDataY


def predict(testDataX, trainDataX, pow, k, trainDataY):
    ans = []
    for element in testDataX:
        result = [0] * 2
        distances = np.array([dist(element, point2, pow) for point2 in trainDataX])
        distances_index = np.argsort(distances)
        #print(distances_index)
        for i in range(k):
            result[int(trainDataY[distances_index[i]][0])] += 1
            #print(result)
        ans.append(np.argmax(result))
    return ans


def calculateAccuracy(predicted, real):
    tp = 0
    tn = 0
    for row, row2 in zip(predicted, real):
        if (row == row2):
            tp += 1
        else:
            tn += 1
    return tp / (tp + tn)


def showResult(X, y):
    # foldNum = [10, 20, 30]
    neighbourNum = list(np.arange(1, 50, 2))
    mRange = range(2, 5)
    plt.gca().set_color_cycle(['red', 'green', 'blue', 'black'])
    legendNames = []
    maxAccur = 0
    maxK = 0
    # for folds in foldNum:
    folderNum = list(range(0, 10))
    for m in mRange:
        accurOnK = []
        for curK in neighbourNum:
            allAccur = 0
            for curFolder in folderNum:
                trainDataX, testDataX, trainDataY, testDataY = splitTrainTest(curFolder, len(X) / len(folderNum), X,
                                                                              y)
                ans = predict(testDataX, trainDataX, m, curK, trainDataY)
                allAccur += calculateAccuracy(ans, testDataY)
            accurOnK.append([curK, allAccur / len(folderNum)])
            if allAccur / len(folderNum) > maxAccur:
                maxAccur = allAccur / len(folderNum)
                maxK = curK
        accurOnK = np.array(accurOnK)
        plt.plot(accurOnK[:, 0],
                 accurOnK[:, 1])
        legendNames.append('M = ' + str(m))
    plt.legend(legendNames, loc='upper left')
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    # plt.title("For " + str(folds) + " folds")
    print("max Accuracy = ", maxAccur, "with k = ", maxK)
    plt.show()


f = open("chips.txt")
data = []
X = []
y = []
for line in f:
    splt_line = line.replace(',', '.').split()
    data.append([float(x) for x in splt_line])
data = np.array(data)
np.random.shuffle(data)
for line in data:
    X.append(line[0: -1])
    y.append(line[-1:])
for row in X:
    row[0:] = [float(c) for c in row[0:]]
for row in y:
    row[0:] = [float(c) for c in row[0:]]
y = np.array(y)
X = np.array(X)
X = normilize(X)
# plt.scatter(data[:, 0], data[:, 1], color=colorData(data))
# center = []
# center.append(sum(data[:, 0]) / len(data[:, 0]))
# center.append(sum(data[:, 1]) / len(data[:, 1]))
# plt.plot(center[0], center[1], "y^")
# plt.show()
# print("Simple coordinates")
# showResult(X, y)
# X_cent = centerShift(X)
# X_polar = toPolar(X_cent)
# plt.scatter(X_polar[:, 0], X_polar[:, 1], color=colorData(data))
# plt.show()
# print("Polar coordinates")
# showResult(X_polar, y)
X_pow2 = np.power(X, 2)
z = [sum(X_pow2[i]) for i in range(len(X))]
z = np.array(z).reshape(len(X), 1)
X_ = np.zeros((len(X), len(X[0]) + 1))
X_[:, :-1] = X
X_[:, -1:] = z
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
curClass = 0
for c, m in [('g', 'o'), ('r', 'o')]:
    kkk = np.where(y == curClass)[0]
    ax.scatter(X_[kkk][:, 0], X_[kkk][:, 1], X_[kkk][:, 2], c=c, marker=m)
    curClass += 1
plt.show()
print("With kernel")
showResult(X_, y)
