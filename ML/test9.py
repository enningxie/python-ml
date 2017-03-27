# 半监督学习
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


# 加载数据集
# 这里使用scikit-learn自带的digits数据集
# 返回的是未标记样本的下标
def load_data():
    digits = datasets.load_digits()
    rng = np.random.RandomState(0)
    indices = np.arange(len(digits.data))  # 样本下标集合
    rng.shuffle(indices)  # 混洗样本下标集合
    x = digits.data[indices]
    y = digits.target[indices]
    # 生成未标记的样本的下标集合
    n_labeled_points = int(len(y)/10)  # 只有10%的样本有标记
    unlabeled_indices = np.arange(len(y))[n_labeled_points:]  # 后面90%的样本未标记
    return x, y, unlabeled_indices


# 标准的迭代式标记传播算法
# 我们需要原始的标记y来评价图半监督学习的效果
# 这里未标记的样本的标记设置为-1
def test_LabelPropagation(*data):
    x, y ,unlabeled_indices = data
    y_train = np.copy(y)  # 这里选择复制，后面要用到y
    y_train[unlabeled_indices] = -1  # 未标记样本的标记设定为-1
    clf = LabelPropagation(max_iter=100, kernel='rbf', gamma=0.1)
    clf.fit(x, y_train)
    # 获取预测准确率
    true_labels = y[unlabeled_indices]  # 取得真实标记
    print("Accuracy: %f" % clf.score(x[unlabeled_indices], true_labels))


# 考虑折中系数alpha以及gamma参数对于rbf核的LabelPropagation的预测性能的影响
def test_LabelPropagation_rbf(*data):
    x, y, unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    alphas = np.linspace(0.01, 1, num=10, endpoint=True)
    gammas = np.logspace(-2, 2, num=50)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
                  , (0, 0.6, 0.4), (0.5, 0.3, 0.2))  # 颜色集合，不同的曲线用不同的颜色
    # 训练并绘图
    for alpha, color in zip(alphas, colors):
        scores = []
        for gamma in gammas:
            clf = LabelPropagation(max_iter=100, gamma=gamma, alpha=alpha, kernel='rbf')
            clf.fit(x, y_train)
            scores.append(clf.score(x[unlabeled_indices], y[unlabeled_indices]))
        ax.plot(gammas, scores, label=r"$\alpha=%s$" % alpha, color=color)
    # 设置图形
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_xscale("log")
    ax.legend(loc='best')
    ax.set_title("LabelPropagation rbf kernel")
    plt.show()


# 考虑折中系数alpha，以及n_neighbors参数对于knn核的LabelPropagation的预测性能的影响
def test_LabelPropagation_knn(*data):
    x, y, unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    alphas = np.linspace(0.01, 1, num=10, endpoint=True)
    Ks = [1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50]
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),\
              (0, 0.6, 0.4), (0.5, 0.3, 0.2))  # 颜色集合，不同的曲线用不同的颜色
    # 训练并绘图
    for alpha, color in zip(alphas, colors):
        scores = []
        for K in Ks:
            clf = LabelPropagation(max_iter=100, n_neighbors=K, alpha=alpha, kernel='knn')
            clf.fit(x, y_train)
            scores.append(clf.score(x[unlabeled_indices], y[unlabeled_indices]))
        ax.plot(Ks, scores, label=r"$\alpha=%s$" % alpha, color=color)
    # 设置图形
    ax.set_xlabel(r"k")
    ax.set_ylabel("score")
    ax.legend(loc='best')
    ax.set_title("LabelPropagation knn kernel")
    plt.show()


# 类图半监督学习算法/该算法对未标记的样本的预测准确率为97%
def test_LabelSpreading(*data):
    x, y, unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    clf = LabelSpreading(max_iter=100, kernel='rbf', gamma=0.1)
    clf.fit(x, y_train)
    predicted_labels = clf.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]
    print("Accuracy: %f" % metrics.accuracy_score(true_labels, predicted_labels))


# 考察折中系数alpha以及gamma参数对于rbf核的LabelSpreading的预测性能的影响
def test_LabelSpreading_rbf(*data):
    x, y, unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    alphas = np.linspace(0.01, 1, num=10, endpoint=True)
    gammas = np.logspace(-2, 2, num=50)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
                  , (0, 0.6, 0.4), (0.5, 0.3, 0.2))  # 颜色集合，不同的曲线用不同的颜色
    # 训练并绘图
    for alpha, color in zip(alphas, colors):
        scores = []
        for gamma in gammas:
            clf = LabelSpreading(max_iter=100, gamma=gamma, alpha=alpha, kernel='rbf')
            clf.fit(x, y_train)
            scores.append(clf.score(x[unlabeled_indices], y[unlabeled_indices]))
        ax.plot(gammas, scores, label=r"$\alpha=%s$" % alpha, color=color)
    # 设置图形
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_xscale("log")
    ax.legend(loc='best')
    ax.set_title("LabelSpreading rbf kernel")
    plt.show()


# 最后考察折中系数alpha以及n_neighbors参数对于knn核的LabelSpreading的预测性能的影响
def test_LabelSpreading_knn(*data):
    x, y, unlabeled_indices = data
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    alphas = np.linspace(0.01, 1, num=10, endpoint=True)
    Ks = [1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50]
    colors = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0), \
    (0, 0.6, 0.4), (0.5, 0.3, 0.2))  # 颜色集合，不同的曲线用不同的颜色
    # 训练并绘图
    for alpha, color in zip(alphas, colors):
        scores = []
        for K in Ks:
            clf = LabelSpreading(max_iter=100, n_neighbors=K, alpha=alpha, kernel='knn')
            clf.fit(x, y_train)
            scores.append(clf.score(x[unlabeled_indices], y[unlabeled_indices]))
        ax.plot(Ks, scores, label=r"$\alpha=%s$" % alpha, color=color)
    # 设置图形
    ax.set_xlabel(r"k")
    ax.set_ylabel("score")
    ax.legend(loc='best')
    ax.set_title("LabelSpreading knn kernel")
    plt.show()

# 总结
# 半监督学习在利用未标记样本后并非必然提高泛化能力，在有些情况下甚至会导致性能下降
# 对于生成方法，原因通常是模型假设不准确。因此需要依赖充分可靠的领域知识来设计模型。
# 更一般的安全半监督学习仍然是为解决的难题，安全是指：利用未标记样本后，能确保返回性能至少不差于仅利用标记样本。
#