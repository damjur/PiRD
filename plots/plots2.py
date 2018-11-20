import matplotlib.pyplot as plt
import numpy as np

algorithms = ["1NN", "3NN", "5NN"]
precision = []
recall = []

f = open("out7x7x2.txt", "r")
for _ in range(60):
    f.readline()
    for i, algorithm in enumerate(algorithms):
        p, r = f.readline().split(' ')
        # print(algorithm, float(p), float(r))
        precision.append(float(p))
        recall.append(float(r))
print(precision)
print(recall)
print(sum(precision)/len(precision), sum(recall)/len(precision))
markers = ["r", "g", "b", "c", "y"]
for i, algorithm in enumerate(algorithms):
    plt.figure(0)
    plt.plot(0, precision[i], markers[i], label=algorithm)
    plt.figure(1)
    plt.plot(0, recall[i], markers[i], label=algorithm)

plt.figure(0)
plt.xlabel('zbiory testowe')
plt.ylabel('precyzja')
plt.legend()
plt.grid(True)
plt.savefig("./precision.pdf")

plt.figure(1)
plt.xlabel('zbiory testowe')
plt.ylabel('czułość')
plt.legend()
plt.grid(True)
plt.savefig("./recall.pdf")
