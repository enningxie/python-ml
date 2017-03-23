import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 给出一个随机产生的数据集
def create_data(n):
    np.random.seed(0)
    x = 5 * np.random.rand(n, 1)
    y = np.sin(x).ravel()
    noise_num = (int)(n/5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))
    # print('x:', x)
    # print('y:', y)
    return train_test_split(x, y, test_size=0.25, random_state=1)


# 测试函数
def test_DecisionTreeRegressor(*data):
    x_train, x_test, y_train, y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(x_train, y_train)
    print("Training score:%f" % regr.score(x_train, y_train))
    print("Testing score:%f" % regr.score(x_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y = regr.predict(X)
    ax.scatter(x_train, y_train, label='train sample', c='g')
    ax.scatter(x_test, y_test, label='test sample', c='r')
    ax.plot(X, Y, label='predict_value', linewidth=2, alpha=0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()


# 检验随机划分与最优划分的影响/最优划分预测能力较强
def test_DecisionTreeRegressor_splitter(*data):
    x_train, x_test, y_train, y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(x_train, y_train)
        print("Splitter:", splitter)
        print("Training score:%f" % regr.score(x_train, y_train))
        print("Testing score:%f" % regr.score(x_test, y_test))


# 考虑决策树的深度，决策树的深度越深对应着树就越复杂。也就是模型越复杂。
def test_DecisionTreeRegressor_depth(*data, maxdepth):
    x_train, x_test, y_train, y_test = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(x_train, y_train)
        training_scores.append(regr.score(x_train, y_train))
        testing_scores.append(regr.score(x_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label='train score')
    ax.plot(depths, testing_scores, label='test score')
    ax.set_xlabel('maxdepth')
    ax.set_ylabel('score')
    ax.set_title('Decision Tree Regression')
    ax.legend(framealpha=0.5)
    plt.show()


# 分类决策树
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


def load_data():
    '''采用xx花的数据集，一共有150个数据，这些数据分为3类，每类50个数据，每个数据包含4个属性'''
    iris = datasets.load_iris()
    x_train = iris.data
    y_train = iris.target
    return cdtrain_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
# 这里采用分层采样。因为原始数据集中，前50个样本都是0，中间50个为1，最后50个为2，如果不采用分层采样，那么最后切分得到的测试集就不无偏了


# 测试分类决策树
def test_DecisionTreeClassifier(*data):
    x_train, x_test, y_train, y_test = data
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    print("Training score:", clf.score(x_train, y_train))
    print("Testing score:", clf.score(x_test, y_test))


# 考虑评价切分质量的评价标准criterion对于分类的影响（gini/entropy）
def test_DecisionTreeClassifier_criterion(*data):
    x_train, x_test, y_train, y_test = data
    criterions = ['gini', 'entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(x_train, y_train)
        print("Criterion:", criterion)
        print("Training score:", clf.score(x_train, y_train))
        print("Testing score:", clf.score(x_test, y_test))
        # 使用gini系数的策略预测性能较高


# 检测随机划分与最优划分的影响
def test_DecisionTreeClassifier_splitter(*data):
    x_train, x_test, y_train, y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(x_train, y_train)
        print("splitter:", splitter)
        print("Training score:", clf.score(x_train, y_train))
        print("Testing score:", clf.score(x_test, y_test))
        # best策略性能好一点


# 考虑树的深度的影响
def test_DecisionTreeClassifier_depth(*data, maxdepth):
    x_train, x_test, y_train, y_test = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(x_train, y_train)
        training_scores.append(clf.score(x_train, y_train))
        testing_scores.append(clf.score(x_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label="train score", marker='o')
    ax.plot(depths, testing_scores, label="test score", marker='*')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('score')
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5, loc='best')
    plt.show()