import numpy as np

a = np.array([90, 85, 77])
print(a)
print("平均:", a.mean())
print("最高:", a.max())
print("最低:", a.min())
print("标准差:", a.std())

b = np.array([[1, 2], [3, 4]])
print(b.shape)
print(b.T)

scores = np.random.randint(0, 101, 10)
print("成绩:", scores)
print("平均:", scores.mean())
print("最高:", scores.max())
print("最低:", scores.min())

import matplotlib.pyplot as plt

scores = [90, 85, 77, 60, 95]
plt.plot(scores, marker="o")
plt.title("成绩趋势")
plt.xlabel("序号")
plt.ylabel("分数")
plt.savefig("chart.png")
plt.show()