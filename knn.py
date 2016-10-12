import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kFold(data, k):
    test_data = []
    train_data = []

    i = 0
    length = (int) (len(data) / k)

    while i < k:
        temp_data = data
        test_data.append(temp_data[i * length : (i + 1) * length])
        train_data.append(np.delete(temp_data, range(i * length, (i + 1) * length), axis = 0))
        i += 1


    return train_data, test_data

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    # return(rho, phi)
    return(phi, rho)

def distance(a, b, m):
    return math.pow(abs(a[0] - b[0]) ** m + abs(a[1] - b[1]) ** m, 1/m)

def classify (train, test, k, classesAmount, m) :
    ans = []
    for testPoint in test:
        #calculate distances between test point and train points
        dist = np.array([distance(testPoint, train[i], m) for i in range(len(train))])
        distIdx = np.argsort(dist)
        res = [0 for i in range(classesAmount)]
        for idx in distIdx[0:k]:
            res[int(train[idx][2])] += 1
        ans.append(np.argmax(res))
    return ans

def measures (classified_data):
    # Float so that automatic integer division doesn't fuck up the results
    tp = tn = fp = fn = 0.0
    for i in range(len(classified_data)):
        # if test_data is 1 (1 is truthy here)
        if test_data[0][i][2]:
            # t_d is 1, so if c_d is 1 too - we have TP, else - FN
            if classified_data[i] == test_data[0][i][2]:
                tp += 1
            else:
                fn += 1
        else:
            # same deal but TN and FP now
            if classified_data[i] == test_data[0][i][2]:
                tn += 1
            else:
                fp += 1

    # print(tp, tn, fp, fn)

    positive_ex = tp + fn
    negative_ex = fp + tn
    recall = tp / positive_ex
    precision = tp / (tp + fp)

    accuracy = (tp + tn) / (positive_ex + negative_ex)
    f1 = 2 * (precision * recall) / (precision + recall)
    return recall, precision, accuracy, f1

data = []
f = open('chips.txt', 'r')
for line in f:
    data.append(line.replace(',', '.').split())
f.close()

data = np.float32(data)

#print(len(data))
color = ['blue', 'red']
color_data = []
for x in data[:, 2]:
    color_data.append(color[int(x)])

#normalize
# x_min = min(data[:, 0])
# x_max = max(data[:, 0])
# y_min = min(data[:, 1])
# y_max = max(data[:, 1])
# print(x_min, x_max, y_min, y_max)
# data[:, 0] = [(x - x_min) / (x_max - x_min) for x in data[:, 0]]
# data[:, 1] = [(y - y_min) / (y_max - y_min) for y in data[:, 1]]

#np.random.shuffle(data)
color_data = []
for x in data[:, 2]:
    color_data.append(color[int(x)])

#print(color_data)
#print(color[np.array(data)[:, 2]])

#print data
# plt.scatter(data[:, 0], data[:, 1], color = color_data)
# plt.show()

# fig = plt.figure()
ax = plt.subplot(111, projection='3d')
z_coords = [vec[0]**2 + vec[1]**2 for _, vec in enumerate(data)]
ax.scatter(data[:, 0], data[:, 1], z_coords, c = color_data)

# polar visualization
# polar_data = [list(cart2pol(point[0], point[1])) for point in data]
# polar_data = np.float32(polar_data)
#
# pol_plot = plt.subplot(111, projection='polar')
# pol_plot.scatter(polar_data[:, 0], polar_data[:, 1], color = color_data)
# pol_plot.grid(True)

plt.show()
#get k-cross validation
train_data, test_data = kFold(data, 10)
#print(test_data, train_data)
#test = np.delete(data, range(0, 4), axis = 0)
#print(len(test))




rc = pc = acc = f1 = 0.0
#calculate average measures
for i in range(len(train_data)):
    classified = classify(train_data[i], test_data[i], 10, 2, 2)
    rc_tmp, pc_tmp, acc_tmp, f1_tmp = measures(classified)
    print(rc_tmp, pc_tmp, acc_tmp, f1_tmp)
    rc += rc_tmp
    pc += pc_tmp
    acc += acc_tmp
    f1 += acc_tmp
rc /= len(train_data)
pc /= len(train_data)
acc /= len(train_data)
f1 /= len(train_data)

print(rc, pc, acc, f1)