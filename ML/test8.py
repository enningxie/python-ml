## 人工神经网络
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D绘图
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# 线性可分数据集的生成算法
def create_data(n):
    '''

    :param n: 正类的样本点数量，也是负类的样本点的数量。总样本点的数量为2n
    :return: 所有样本点组成的数组，形状为(2*n, 4)，数组中一行代表一个样本点由其特征和标记组成
    '''
    np.random.seed(1)
    x_11 = np.random.randint(0, 100, (n, 1))
    x_12 = np.random.randint(0, 100, (n, 1, ))
    x_13 = 20 + np.random.randint(0, 100, (n, 1, ))
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = np.random.randint(0, 100, (n, 1))
    x_23 = 10 - np.random.randint(0, 100, (n, 1, ))

    new_x_12 = x_12 * np.sqrt(2) / 2 - x_13 * np.sqrt(2) / 2  # 沿x轴旋转45°
    new_x_13 = x_12 * np.sqrt(2) / 2 + x_13 * np.sqrt(2) / 2  # 沿x轴旋转45°

    new_x_22 = x_22 * np.sqrt(2) / 2 - x_23 * np.sqrt(2) / 2  # 沿x轴旋转45°
    new_x_23 = x_22 * np.sqrt(2) / 2 + x_23 * np.sqrt(2) / 2  # 沿x轴旋转45°

    plus_samples = np.hstack([x_11, new_x_12, new_x_13, np.ones((n, 1))])
    minus_samples = np.hstack([x_21, new_x_22, new_x_23, -np.ones((n, 1))])
    samples = np.vstack([plus_samples, minus_samples])
    np.random.shuffle(samples)  # 混洗数据
    return samples


# 绘制数据集的函数
def plot_samples(ax, samples):
    '''

    :param ax: 一个Axes3D实例，负责绘制图形；
    :param sample: 代表训练数据集的数组，形状为(N, n_features+1)，其中N为样本点的个数,n_features代表特征数
    :return:
    '''
    y = samples[:, -1]
    position_p = y == 1  # 正类的位置
    position_m = y == -1  # 负类的位置
    ax.scatter(samples[position_p, 0], samples[position_p, 1], samples[position_p, 2], marker='+', label='+', color='g')
    ax.scatter(samples[position_m, 0], samples[position_m, 1], samples[position_m, 2], marker='^', label='-', color='y')


# 生成数据和绘制数据的合成
def test(func):
    fig = plt.figure()
    ax = Axes3D(fig)
    data = func(100)
    plot_samples(ax, data)
    ax.legend(loc='best')
    plt.show()


# 感知机学习算法的原始形式
def perceptron(train_data, eta, w_0, b_0):
    '''

    :param train_data: 代表训练数据集的数组，形式为(N, n_features+1)
    :param eta: 学习率
    :param w_0: 一个列向量
    :param b_0: 一个标量
    :return: 一个元组成员为w,b以及迭代次数
    '''
    x = train_data[:, :-1]  # x数据
    y = train_data[:, -1]  # 对应的分类
    length = train_data.shape[0] # 样本集大小
    w = w_0
    b = b_0
    step_num = 0
    while True:
        i = 0
        while(i < length):  # 遍历一轮样本集中所有的样本点
            step_num += 1
            x_i = x[i].reshape((x.shape[1], 1))
            y_i = y[i]
            if y_i * (np.dot(np.transpose(w), x_i) + b) <= 0:  # 该点是误分类点
                w = w+eta*y_i*x_i  # 梯度下降
                b = b+eta*y_i
                break  # 执行下一轮筛选
            else:  # 该点不是误分类点，选取下一个样本点
                i = i+1
        if(i == length):  # 没有误分类点，结束循环
            break
    return (w, b, step_num)


# 分离超平面函数
def create_hyperplane(x, y, w, b):
    '''

    :param x: 分离超平面上的点的x坐标组成的数组
    :param y: 分离超平面上的点的y坐标组成的数组
    :param w:即w，超平面的法向量，它是一个列向量
    :param b:即b，超平面的截距
    :return:分离超平面上的点的z坐标组成的数组
    '''
    return (-w[0][0]*x-w[1][0]*y-b)/w[2][0]


# 感知机原始算法的运行情况
def test_perceptron():
    data = create_data(100)
    eta, w_0, b_0 = 0.1, np.ones((3, 1), dtype=float), 1
    w, b, num = perceptron(data, eta, w_0, b_0)
    fig = plt.figure()
    plt.suptitle("perceptron")
    ax = Axes3D(fig)
    # 绘制样本点
    plot_samples(ax, data)
    # 绘制分离超平面
    x = np.linspace(-30, 100, 100)  # 分离超平面的x坐标数组
    y = np.linspace(-30, 100, 100)  # 分离超平面的y坐标数组
    x, y = np.meshgrid(x, y)  # 划分网格
    z = create_hyperplane(x, y, w, b)  # 分离超平面的z坐标数组
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='g', alpha=0.2)
    ax.legend(loc='best')
    plt.show()


# 感知机学习算法的对偶形式
def create_w(train_data, alpha):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    N = train_data.shape[0]
    w = np.zeros((x.shape[1], 1))
    for i in range(0, N):
        w = w+alpha[i][0]*y[i]*(x[i].reshape(x[i].size, 1))
    return w


def perceptron_dual(train_data, eta, alpha_0, b_0):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    length = train_data.shape[0]
    alpha = alpha_0
    b = b_0
    step_num = 0
    while True:
        i = 0
        while(i < length):
            step_num += 1
            x_i = x[i].reshape((x.shape[1], 1))
            y_i = y[i]
            w = create_w(train_data, alpha)
            z = y_i*(np.dot(np.transpose(w), x_i)+b)
            if z <= 0:
                alpha[i][0] += eta
                b += eta*y_i
                break
            else:
                i += 1
        if i == length:
            break
    return alpha, b, step_num


def test_perceptron_dual():
    data = create_data(100)
    eta, w_0, b_0 = 0.1, np.ones((3, 1), dtype=float), 1
    w_1, b_1, num_1 = perceptron(data, eta, w_0, b_0)
    alpha, b_2, num_2 = perceptron_dual(data, eta=0.1, alpha_0=np.zeros((data.shape[0]*2, 1)), b_0=0)
    w_2 = create_w(data, alpha)
    print("w_1, b_1", w_1, b_1)
    print("w_2, b_2", w_2, b_2)
    fig = plt.figure()
    plt.suptitle("perceptron")
    ax = Axes3D(fig)
    plot_samples(ax, data)
    x = np.linspace(-30, 100, 100)  # 分离超平面的x坐标数组
    y = np.linspace(-30, 100, 100)  # 分离超平面的y坐标数组
    x, y = np.meshgrid(x, y)  # 划分网格
    z = create_hyperplane(x, y, w_1, b_1)  # 分离超平面的z坐标数组
    z_2 = create_hyperplane(x, y, w_2, b_2)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='g', alpha=0.2)
    ax.plot_surface(x, y, z_2, rstride=1, cstride=1, color='c', alpha=0.2)
    ax.legend(loc='best')
    plt.show()
# 分离超平面的位置是由少部分重要的样本点决定的，而感知机学习算法的对偶形式能够找出这些重要的样本点，这就是支持向量机的原理


#  学习率与收敛速度
def test_eta(data, ax, etas, w_0, alpha_0, b_0):
    nums1 = []
    nums2 = []
    for eta in etas:
        _, _, num_1 = perceptron(data, eta, w_0=w_0, b_0=b_0)
        _, _, num_2 = perceptron_dual(data, eta=0.1, alpha_0=alpha_0, b_0=b_0)
        nums1.append(num_1)
        nums2.append(num_2)
    ax.plot(etas, nums1, label='orignal')
    ax.plot(etas, nums2, label='dual')


def test_eta_():  # 实验表明，不同的eta对学习算法的收敛速度有影响，但是影响不大
    fig = plt.figure()
    fig.suptitle("eta_perceptron")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r"$\eta$")

    data = create_data(20)
    etas = np.linspace(0.01, 1, num=25, endpoint=False)
    w_0, b_0, alpha_0 = np.ones((3, 1)), 0, np.zeros((data.shape[0], 1))
    test_eta(data, ax, etas, w_0, alpha_0, b_0)
    ax.legend(loc='best', framealpha=0.5)
    plt.show()


# 感知机与线性不可分数据集
# 生成线性不可分数据集/与上述生成可分数据集函数不同之处在于随机数区间有所变化导致正类和负类样本相互交叉。
def create_data_no_linear(n):
    np.random.seed(1)
    x_11 = np.random.randint(0, 100, (n, 1))
    x_12 = np.random.randint(0, 100, (n, 1,))
    x_13 = 10 + np.random.randint(0, 10, (n, 1,))
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = np.random.randint(0, 100, (n, 1))
    x_23 = 20 - np.random.randint(0, 10, (n, 1,))

    new_x_12 = x_12 * np.sqrt(2) / 2 - x_13 * np.sqrt(2) / 2  # 沿x轴旋转45°
    new_x_13 = x_12 * np.sqrt(2) / 2 + x_13 * np.sqrt(2) / 2  # 沿x轴旋转45°

    new_x_22 = x_22 * np.sqrt(2) / 2 - x_23 * np.sqrt(2) / 2  # 沿x轴旋转45°
    new_x_23 = x_22 * np.sqrt(2) / 2 + x_23 * np.sqrt(2) / 2  # 沿x轴旋转45°

    plus_samples = np.hstack([x_11, new_x_12, new_x_13, np.ones((n, 1))])
    minus_samples = np.hstack([x_21, new_x_22, new_x_23, -np.ones((n, 1))])
    samples = np.vstack([plus_samples, minus_samples])
    np.random.shuffle(samples)  # 混洗数据
    return samples


# 感知机学习算法只能够用于线性可分数据集
# 多层神经网络
# 生成线性不可分的二维特征数据
def create_data_no_linear_2d(n):
    np.random.seed(1)
    x_11 = np.random.randint(0, 100, (n, 1))
    x_12 = 10+np.random.randint(-5, 5, (n, 1,))
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = 20+np.random.randint(0, 10, (n, 1))
    x_31 = np.random.randint(0, 100, (int(n/10), 1))
    x_32 = 20+np.random.randint(0, 10, (int(n/10), 1))

    new_x_11 = x_11 * np.sqrt(2) / 2 - x_12 * np.sqrt(2) / 2
    new_x_12 = x_11 * np.sqrt(2) / 2 + x_12 * np.sqrt(2) / 2
    new_x_21 = x_21 * np.sqrt(2) / 2 - x_22 * np.sqrt(2) / 2
    new_x_22 = x_21 * np.sqrt(2) / 2 + x_22 * np.sqrt(2) / 2
    new_x_31 = x_31 * np.sqrt(2) / 2 - x_32 * np.sqrt(2) / 2
    new_x_32 = x_31 * np.sqrt(2) / 2 + x_32 * np.sqrt(2) / 2

    plus_samples = np.hstack([new_x_11, new_x_12, np.ones((n, 1))])
    minus_samples = np.hstack([new_x_21, new_x_22, -np.ones((n, 1))])
    err_samples = np.hstack([new_x_31, new_x_32, np.ones((int(n/10), 1))])
    samples = np.vstack([plus_samples, minus_samples, err_samples])
    np.random.shuffle(samples)
    return samples


# 绘制
def plot_samples_2d(ax, samples):
    y = samples[:, -1]
    position_p = y == 1
    position_m = y == -1
    ax.scatter(samples[position_p, 0], samples[position_p, 1], marker='+', label='+', color='b')
    ax.scatter(samples[position_m, 0], samples[position_m, 1], marker='^', label='-', color='y')


def test_2d():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = create_data_no_linear_2d(100)
    plot_samples_2d(ax, data)
    ax.legend(loc='best')
    plt.show()


# 使用多层神经网络MLPClassifier来处理非线性数据集
def predict_with_MLPClassifier(ax, train_data):
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    clf = MLPClassifier(activation='logistic', max_iter=1000)
    clf.fit(train_x, train_y)
    print(clf.score(train_x, train_y))

    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
    plot_step = 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=plt.cm.Paired)


def test_MLPClassifier():
    data = create_data_no_linear_2d(500)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    predict_with_MLPClassifier(ax, data)
    plot_samples_2d(ax, data)
    ax.legend(loc='best')
    plt.show()


# 多层神经网络的应用
# 对xx花进行分类，xx花数据集一共有150个数据，这些数据分为3类，每类50个数据，每个数据包含4个属性
# 加载数据集
def load_data_MLP():
    np.random.seed(0)
    iris = load_iris()
    x = iris.data[:, 0:2]
    y = iris.target
    data = np.hstack((x, y.reshape(y.size, 1)))
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1]
    train_x = x[:-30]
    test_x = x[-30:]
    train_y = y[:-30]
    test_y = y[-30:]
    return train_x, test_x, train_y, test_y, iris


def mlpclassifier_iris():
    train_x, test_x, train_y, test_y, iris= load_data_MLP()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    classifier = MLPClassifier(activation='logistic', max_iter=10000, hidden_layer_sizes=(30,))
    classifier.fit(train_x, train_y)
    train_score = classifier.score(train_x, train_y)
    test_score = classifier.score(test_x, test_y)
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
    plot_classifier_predict_meshgrid(ax, classifier, x_min, x_max, y_min, y_max)
    plot_samples(ax, train_x, train_y)
    ax.legend(loc='best')
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title("train_score: %f;test_score: %f" % (train_score, test_score))
    plt.show()


# 绘制样本点
def plot_samples(ax, x, y):
    iris = load_iris()
    n_classes = 3
    plot_colors = 'bry'
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(x[idx, 0], x[idx, 1], c=color, label=iris.target_names[i], cmap=plt.cm.Paired)


def plot_classifier_predict_meshgrid(ax, clf, x_min, x_max, y_min, y_max):
    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=plt.cm.Paired)


# 不同的隐含层对于多层神经网络分类器的影响/试错法，解决隐含层选取问题
def mlpclassifier_iris_hidden_layer_sizes():
    train_x, test_x, train_y, test_y, iris = load_data_MLP()
    fig = plt.figure()
    hidden_layer_sizes = [(10,), (30,), (100,), (5, 5), (10, 10), (30, 30)]
    for itx, size in enumerate(hidden_layer_sizes):
        ax = fig.add_subplot(2, 3, itx+1)
        classifier = MLPClassifier(activation='logistic', max_iter=10000, hidden_layer_sizes=size)
        classifier.fit(train_x, train_y)
        train_score = classifier.score(train_x, train_y)
        test_score = classifier.score(test_x, test_y)
        x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
        y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
        plot_classifier_predict_meshgrid(ax, classifier, x_min, x_max, y_min, y_max)
        plot_samples(ax, train_x, train_y)
        ax.legend(loc='best')
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title("layer_size: %s;train_score: %f;test_score: %f" % (size, train_score, test_score))
    plt.show()


# 考虑到激活函数的影响/实验表明，不同的激活函数对此数据集的分类效果差不多
def mlpclassifier_iris_ativations():
    train_x, test_x, train_y, test_y, iris = load_data_MLP()
    fig = plt.figure()
    activations = ['logistic', 'tanh', 'relu']
    for itx, act in enumerate(activations):
        ax = fig.add_subplot(1, 3, itx + 1)
        classifier = MLPClassifier(activation=act, max_iter=10000, hidden_layer_sizes=(30,))
        classifier.fit(train_x, train_y)
        train_score = classifier.score(train_x, train_y)
        test_score = classifier.score(test_x, test_y)
        x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
        y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
        plot_classifier_predict_meshgrid(ax, classifier, x_min, x_max, y_min, y_max)
        plot_samples(ax, train_x, train_y)
        ax.legend(loc='best')
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title("activation: %s;train_score: %f;test_score: %f" % (act, train_score, test_score))
    plt.show()


# 考虑优化算法
def mlpclassifier_iris_algorithms():
    train_x, test_x, train_y, test_y, iris = load_data_MLP()
    fig = plt.figure()
    algorithms = ['lbfgs', 'sgd', 'adam']
    for itx, alg in enumerate(algorithms):
        ax = fig.add_subplot(1, 3, itx + 1)
        classifier = MLPClassifier(activation='tanh', max_iter=10000, hidden_layer_sizes=(30,), solver=alg)
        classifier.fit(train_x, train_y)
        train_score = classifier.score(train_x, train_y)
        test_score = classifier.score(test_x, test_y)
        x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
        y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
        plot_classifier_predict_meshgrid(ax, classifier, x_min, x_max, y_min, y_max)
        plot_samples(ax, train_x, train_y)
        ax.legend(loc='best')
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title("algorithm: %s;train_score: %f;test_score: %f" % (alg, train_score, test_score))
    plt.show()


# 最后考虑学习效率的影响/并不是eta越小越好
def mlpclassifier_iris_eta():
    train_x, test_x, train_y, test_y, iris = load_data_MLP()
    fig = plt.figure()
    etas = [0.1, 0.01, 0.001, 0.0001]
    for itx, eta in enumerate(etas):
        ax = fig.add_subplot(2, 2, itx + 1)
        classifier = MLPClassifier(activation='tanh', max_iter=10000, hidden_layer_sizes=(30,), learning_rate_init=eta, solver='sgd')
        classifier.fit(train_x, train_y)
        train_score = classifier.score(train_x, train_y)
        test_score = classifier.score(test_x, test_y)
        x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
        y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
        plot_classifier_predict_meshgrid(ax, classifier, x_min, x_max, y_min, y_max)
        plot_samples(ax, train_x, train_y)
        ax.legend(loc='best')
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title(r"$\eta: %s;train_score: %f;test_score: %f$" % (eta, train_score, test_score))
    plt.show()