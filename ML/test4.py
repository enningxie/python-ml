# k近邻算法
import numpy as np
import matplotlib.pylab as plt
from sklearn import neighbors, datasets
from sklearn import model_selection


# 加载用于分类的数据集
def load_classification_data():
    ''' 数据集合由1797张样本图片组成 '''
    digits = datasets.load_digits()
    return model_selection.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# 加载用于回归的数据集
def load_regression_data(n):
    ''' 在sin(x)基础上添加噪声生成的 '''
    x = 5 * np.random.rand(n, 1)
    y = np.sin(x).ravel()
    y[::5] += 1 * (0.5 - np.random.rand(int(n/5)))
    return model_selection.train_test_split(x, y, test_size=0.25, random_state=0)


# k近邻，权值和k对score的影响/distance策略表现较好
def test_KNeighborsClassifier_k_w(*data):
    x_train, x_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for weight in weights:
        training_scores = []
        testing_scores = []
        for K in Ks:
            clf = neighbors.KNeighborsClassifier(weights=weight, n_neighbors=K)
            clf.fit(x_train, y_train)
            training_scores.append(clf.score(x_train, y_train))
            testing_scores.append(clf.score(x_test, y_test))
        ax.plot(Ks, training_scores, label='train score(%s)' % weight)
        ax.plot(Ks, testing_scores, label='test score(%s)' % weight)
    ax.legend(loc='best')
    ax.set_xlabel('K')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.05)
    ax.set_title('KNeighborsClassifier')
    plt.show()


# 考虑不同的距离度量对预测性能的影响/p参数对性能预测影响不大，不同的距离度量作用相似
def test_KNeighborsClassifier_k_p(*data):
    x_train, x_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    Ps = [1, 2, 10]
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for P in Ps:
        train_scores = []
        test_scores = []
        for K in Ks:
            clf = neighbors.KNeighborsClassifier(p=P, n_neighbors=K)
            clf.fit(x_train, y_train)
            train_scores.append(clf.score(x_train, y_train))
            test_scores.append(clf.score(x_test, y_test))
        ax.plot(Ks, train_scores, label='train score(%d)' % P)
        ax.plot(Ks, test_scores, label='test scpre(%d)' % P)
    ax.legend(loc='best')
    ax.set_xlabel('K')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.05)
    ax.set_title('KNeighborsClassifier')
    plt.show()


# k 回归
def test_KNeighborsRegressor(*data):
    x_train, x_test, y_train, y_test = data
    regr = neighbors.KNeighborsRegressor()
    regr.fit(x_train, y_train)
    print("Training score:", regr.score(x_train, y_train))
    print("Test score:", regr.score(x_test, y_test))


# 考虑k值和权值对预测的影响
def test_KNeighborsRegressor_k_w(*data):
    x_train, x_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for weight in weights:
        training_scores = []
        testing_scores = []
        for K in Ks:
            regr = neighbors.KNeighborsRegressor(weights=weight, n_neighbors=K)
            regr.fit(x_train, y_train)
            training_scores.append(regr.score(x_train, y_train))
            testing_scores.append(regr.score(x_test, y_test))
        ax.plot(Ks, training_scores, label='train score(%s)' % weight)
        ax.plot(Ks, testing_scores, label='test score(%s)' % weight)
    ax.legend(loc='best')
    ax.set_xlabel('K')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.05)
    ax.set_title('KNeighborsRegressor')
    plt.show()


# 考虑不同的距离度量对预测性能的影响/p参数对性能预测影响不大，不同的距离度量作用相似
def test_KNeighborsRegressor_k_p(*data):
    x_train, x_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    Ps = [1, 2, 10]
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for P in Ps:
        train_scores = []
        test_scores = []
        for K in Ks:
            regr = neighbors.KNeighborsRegressor(p=P, n_neighbors=K)
            regr.fit(x_train, y_train)
            train_scores.append(regr.score(x_train, y_train))
            test_scores.append(regr.score(x_test, y_test))
        ax.plot(Ks, train_scores, label='train score(%d)' % P)
        ax.plot(Ks, test_scores, label='test scpre(%d)' % P)
    ax.legend(loc='best')
    ax.set_xlabel('K')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.05)
    ax.set_title('KNeighborsClassifier')
    plt.show()