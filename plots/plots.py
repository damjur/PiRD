import matplotlib.pyplot as plt
import numpy as np

algorithms = ["1NN", "3NN", "5NN", "min", "avg"]
precision = [[] for _ in range(len(algorithms))]
recall = [[] for _ in range(len(algorithms))]
tests = ["0/6", "1/6", "2/6", "3/6"]
f = open("results13x26.txt", "r")
for test in tests:
    for i, algorithm in enumerate(algorithms):
        p, r = f.readline().split('\t')
        # print(algorithm, float(p), float(r))
        precision[i].append(float(p))
        recall[i].append(float(r))
# print(precision)
# print(recall)

markers = ["r", "g", "b", "c", "y"]
for i, algorithm in enumerate(algorithms):
    plt.figure(0)
    plt.plot(tests, precision[i], markers[i], label=algorithm)
    plt.figure(1)
    plt.plot(tests, recall[i], markers[i], label=algorithm)

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
