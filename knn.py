import matplotlib.pyplot as plt
import numpy as np
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
x_min = min(data[:, 0])
x_max = max(data[:, 0])
y_min = min(data[:, 1])
y_max = max(data[:, 1])
print(x_min, x_max, y_min, y_max)
data[:, 0] = [(x - x_min) / (x_max - x_min) for x in data[:, 0]]
data[:, 1] = [(y - y_min) / (y_max - y_min) for y in data[:, 1]]


#print(color_data)
#print(color[np.array(data)[:, 2]])

#print data
plt.scatter(data[:, 0], data[:, 1], color = color_data)
plt.show()

np.random.shuffle(data)
color_data = []
for x in data[:, 2]:
    color_data.append(color[int(x)])
plt.scatter(data[:, 0], data[:, 1], color = color_data)
plt.show()