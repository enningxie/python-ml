# 贝叶斯分类器
from sklearn import datasets, naive_bayes
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# 这里使用的数据集是scikit-learn自带的手写数字识别数据集Digit Dataset。
def show_digits():
    '''
    该数据集由1797张样本图片组成，每张图片都是一个8×8大小的手写数字位图。
    为了便于处理，scikit-learn将样本图片转换成64维的向量。
    '''
    digits = datasets.load_digits()
    fig = plt.figure()
    print("vector from images 0:", digits.data[0])
    for i in range(25):
        ax = fig.add_subplot(5, 5, i+1)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


# 加载数据集的函数
def load_data():
    digits = datasets.load_digits()
    return train_test_split(digits.data, digits.target, test_size=0.25, random_state=0, stratify=digits.data)


# 高斯贝叶斯分类器（GaussianNB）
def test_GaussianNB(*data):
    x_train, x_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()
    cls.fit(x_train, y_train)
    print("training score:", cls.score(x_train, y_train))
    print("testing score:", cls.score(x_test, y_test))


# 多项式贝叶斯分类器（MultinomialNB）
def test_MultinomialNB(*data):
    x_train, x_test, y_train, y_test = data
    cls = naive_bayes.MultinomialNB()
    cls.fit(x_train, y_train)
    print("training score:", cls.score(x_train, y_train))
    print("testing score:", cls.score(x_test, y_test))


# 不同的alpha值对于多项式贝叶斯分类的影响，这里的alpha就是贝叶斯估计中的lameda
def test_MultinomialNB_alpha(*data):
    x_train, x_test, y_train, y_test = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, train_scores, label="train score")
    ax.plot(alphas, test_scores, label="test score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)  # 限定y轴范围
    ax.set_title("MultinomialNB")
    ax.set_xscale("log")  # x轴指数表示
    ax.legend(framealpha=0.5)
    plt.show()


# 伯努利贝叶斯分类器（BernoulliNB）
def test_BernoulliNB(*data):
    x_train, x_test, y_train, y_test = data
    cls = naive_bayes.BernoulliNB()
    cls.fit(x_train, y_train)
    print("training score:", cls.score(x_train, y_train))
    print("testing score:", cls.score(x_test, y_test))


# 检测不同alpha对于伯努利贝叶斯分类器的影响
def test_BernoulliNB_alpha(*data):
    x_train, x_test, y_train, y_test = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.BernoulliNB(alpha=alpha)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, train_scores, label="train score")
    ax.plot(alphas, test_scores, label="test score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)  # 限定y轴范围
    ax.set_title("BernoulliNB")
    ax.set_xscale("log")  # x轴指数表示
    ax.legend(framealpha=0.5)
    plt.show()


# 考虑binarize的参数对伯努利贝叶斯分类器的影响
def test_BernoulliNB_binarize(*data):
    x_train, x_test, y_train, y_test = data
    min_x = min(np.min(x_train.ravel()), np.min(x_test.ravel())) - 0.1
    max_x = max(np.max(x_train.ravel()), np.max(x_test.ravel())) + 0.1
    binarizes = np.linspace(min_x, max_x, endpoint=True, num=100)
    train_scores = []
    test_scores = []
    for binarize in binarizes:
        cls = naive_bayes.BernoulliNB(binarize=binarize)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append((cls.score(x_test, y_test)))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(binarizes, train_scores, label="train score")
    ax.plot(binarizes, test_scores, label="test score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)  # 限定y轴范围
    ax.set_title("BernoulliNB")
    # ax.set_xscale("log")  # x轴指数表示
    ax.legend(framealpha=0.5)
    plt.show()