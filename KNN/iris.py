from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 导入鸢尾花数据集
iris = datasets.load_iris()
data = iris.data[:, :2]  # 前2列->2个特征
target = iris.target
print('target=', np.unique(target))

# 区分训练集和测试集，75%的训练集和25%的测试集
train_data, test_data = train_test_split(np.c_[data, target])  # test_size自动设置成0.25

# 训练并预测，其中选取k=15
clf = neighbors.KNeighborsClassifier(15, 'distance')
clf.fit(train_data[:, :2], train_data[:, 2])  # X=[:, :2], y=[:, 2]
Z = clf.predict(test_data[:, :2])
print('准确率:', clf.score(test_data[:, :2], test_data[:, 2]))  # X=[:, :2], y=[:, 2]

colormap = dict(zip(np.unique(target), sns.color_palette()[:3]))
plt.scatter(train_data[:, 0], train_data[:, 1], edgecolors=[colormap[x] for x in train_data[:, 2]], c='white', s=80, label='all_data')
plt.scatter(test_data[:, 0], test_data[:, 1], marker='^', color=[colormap[x] for x in Z], s=20, label='test_data')
plt.legend()  # 给图加上图例，默认右上角
plt.show()
