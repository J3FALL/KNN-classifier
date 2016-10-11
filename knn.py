import matplotlib.pyplot as plt
import numpy as np
import math
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

def euDist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def classify (train, test, k, classesAmount) :
    ans = []
    for testPoint in test:
        #calculate distances between test point and train points
        dist = np.array([euDist(testPoint, train[i]) for i in range(len(train))])
        distIdx = np.argsort(dist)
        res = [0 for i in range(classesAmount)]
        for idx in distIdx[0:k]:
            res[int(train[idx][2])] += 1
        ans.append(np.argmax(res))
    return ans

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

np.random.shuffle(data)
color_data = []
for x in data[:, 2]:
    color_data.append(color[int(x)])

#print(color_data)
#print(color[np.array(data)[:, 2]])

#print data
# plt.scatter(data[:, 0], data[:, 1], color = color_data)
# plt.show()

# polar visualization
polar_data = [list(cart2pol(point[0], point[1])) for point in data]
polar_data = np.float32(polar_data)

pol_plot = plt.subplot(111, projection='polar')
pol_plot.scatter(polar_data[:, 0], polar_data[:, 1], color = color_data)
pol_plot.grid(True)
plt.show()
#get k-cross validation
train_data, test_data = kFold(data, 10)
#print(test_data, train_data)
#test = np.delete(data, range(0, 4), axis = 0)
#print(len(test))

print(test_data[0][0], test_data[0][1])
print(euDist(test_data[0][0], test_data[0][1]))

print(classify(train_data[0], test_data[0], 10, 2))

tmp = [int(i[2]) for i in test_data[0]]
print(tmp)