# 降维
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold


def load_data():
    iris = datasets.load_iris()
    # print("x : ", iris.data.shape)
    # print("y : ", iris.target)
    return iris.data, iris.target


def test_PCA(*data):
    x, y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(x)
    # print("主成分数组：", pca.components_)
    # print("主成分元素值：", pca.n_components_)
    # print("主成分特征的统计平均值：", pca.mean_)
    print("explained variance ratio: %s" % str(pca.explained_variance_ratio_))


# 降维后样本分布图函数
def plot_PCA(*data):
    x, y = data
    pca = decomposition.PCA(n_components=2)  # 经过主成分explained variance数组的分析确定的参数2
    pca.fit(x)
    x_r = pca.transform(x)  # 降维操作
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0)\
              , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    for label, color in zip(np.unique(y), colors):
        position = y == label  # 这里的position类型为bool类型
        ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()


# IncrementalPCA 可适用于超大规模数据，可以将数据分批加载进内存中。
# KernelPCA 核化PCA模型
def test_KPCA(*data):
    x, y = data
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']  # 分别为线性核/多项式核/高斯核/..
    for kernel in kernels:
        kpca = decomposition.KernelPCA(n_components=None, kernel=kernel)
        kpca.fit(x)
        print("kernel=%s --> lamdas: %s" % (kernel, kpca.lambdas_))  # 参数lamdas为核化矩阵的特征值


# 给出绘制降维后的样本分布图/不同核函数降维后的分布是不同的
def plot_KPCA(*data):
    x, y = data
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig = plt.figure()
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
                  , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    for i, kernel in enumerate(kernels):
        kpca = decomposition.KernelPCA(n_components=2, kernel=kernel)
        kpca.fit(x)
        x_r = kpca.transform(x)
        ax = fig.add_subplot(2, 2, i+1)
        for label, color in zip(np.unique(y), colors):
            position = y == label  # 这里的position类型为bool类型
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("y[0]")
        ax.legend(loc="best")
        ax.set_title("kernel=%s" % kernel)
    plt.suptitle("KPCA")
    plt.show()


# 考察多项式核的参数的影响
def plot_KPCA_poly(*data):
    x, y = data
    fig = plt.figure()
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
                  , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    Params = [(3, 1, 1), (3, 10, 1), (3, 1, 10), (3, 10, 10), (10, 1, 1), (10, 10, 1), (10, 1, 10), (10, 10, 10)]
    for i, (p, gamma, r) in enumerate(Params):
        kpca = decomposition.KernelPCA(n_components=2, kernel='poly', gamma=gamma, degree=p, coef0=r)
        kpca.fit(x)
        x_r = kpca.transform(x)
        ax = fig.add_subplot(2, 4, i+1)
        for label, color in zip(np.unique(y), colors):
            position = y == label  # 这里的position类型为bool类型
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title(r"$(%s (x \cdot z+1)+%s)^{%s}$" % (gamma, r, p))
    plt.suptitle("KPCA-Poly")
    plt.show()


# 考察高斯核的参数影响
def plot_KPCA_rbf(*data):
    x, y = data
    fig = plt.figure()
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
                  , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    Gammas = [0.5, 1, 4, 10]
    for i, gamma in enumerate(Gammas):
        kpca = decomposition.KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
        kpca.fit(x)
        x_r = kpca.transform(x)
        ax = fig.add_subplot(2, 2, i+1)
        for label, color in zip(np.unique(y), colors):
            position = y == label  # 这里的position类型为bool类型
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title(r"$\exp(-%s||x-z||^2)$" % gamma)
    plt.suptitle("KPCA-rbf")
    plt.show()


# 考察sigmoid核的参数的影响/这里的r参数的选取要小心。
def plot_KPCA_sigmoid(*data):
    x, y = data
    fig = plt.figure()
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
                  , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    Params = [(0.01, 0.1), (0.01, 0.2), (0.1, 0.1), (0.1, 0.2), (0.2, 0.1), (0.2, 0.2)]
    for i, (gamma, r) in enumerate(Params):
        kpca = decomposition.KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma, coef0=r)
        kpca.fit(x)
        x_r = kpca.transform(x)
        ax = fig.add_subplot(3, 2, i+1)
        for label, color in zip(np.unique(y), colors):
            position = y == label  # 这里的position类型为bool类型
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_xticks([])  # 修改轴的刻度
        ax.set_yticks([])
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title(r"$\tanh(%s(x\cdot z)+%s)$" % (gamma, r))
    plt.suptitle("KPCA-sigmoid")
    plt.show()


# 多维缩放模型
def test_MDS(*data):
    x, y = data
    for n in [4, 3, 2, 1]:
        mds = manifold.MDS(n_components=n)
        mds.fit(x)
        print('stress(n_components=%d) : %s' % (n, str(mds.stress_)))  # stress_一个浮点数，给出了不一致的距离的总和
# 该指标并不能用于判定降维效果的好坏，只是一个中性指标


# 描绘降维后的样本分布图
def plot_MDS(*data):
    x, y = data
    mds =manifold.MDS(n_components=2)
    x_r = mds.fit_transform(x)  # 训练模型后并返回低维坐标
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
                  , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    for label, color in zip(np.unique(y), colors):
        position = y == label  # 这里的position类型为bool类型
        ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.legend(loc="best")
    ax.set_title('MDS')
    plt.show()


# Isomap 模型/对于不同的低维空间，其降维的重构误差比较小，中性指标
def test_Isomap(*data):
    x, y = data
    for n in [4, 3, 2, 1]:
        isomap = manifold.Isomap(n_components=n)
        isomap.fit(x)
        print('reconstruction_error(n_component=%d) : %s' % (n, isomap.reconstruction_error()))  # 计算重构误差


# 绘制降维后的样本分布图/k=1出现了断路的现象
def plot_Isomap_k(*data):
    x, y = data
    Ks = [1, 5, 25, y.size-1]
    fig = plt.figure()
    for i, k in enumerate(Ks):
        isomap = manifold.Isomap(n_components=2, n_neighbors=k)  # n_neighbors 近邻参数k
        x_r = isomap.fit_transform(x)
        ax = fig.add_subplot(2, 2, i+1)
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
            , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label  # 这里的position类型为bool类型
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title('K=%d' % k)
    plt.suptitle('Isomap')
    plt.show()


# 最后给出将原始数据直接压缩至一维的情况
def plot_Isomap_k_d1(*data):
    x, y = data
    Ks = [1, 5, 25, y.size-1]
    fig = plt.figure()
    for i, k in enumerate(Ks):
        isomap = manifold.Isomap(n_components=1, n_neighbors=k)  # n_neighbors 近邻参数k
        x_r = isomap.fit_transform(x)
        ax = fig.add_subplot(2, 2, i + 1)
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
            , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label  # 这里的position类型为bool类型
            ax.scatter(x_r[position], np.zeros_like(x_r[position]), label="target=%d" % label, color=color)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="best")
        ax.set_title('K=%d' % k)
    plt.suptitle('Isomap')
    plt.show()


# LLE模型/对于不同的低维空间，其降维的重构误差很小/中性指标
def test_LocallyLinearEmbedding(*data):
    x, y = data
    for n in [4, 3, 2, 1]:
        lle = manifold.LocallyLinearEmbedding(n_components=n)
        lle.fit(x)
        print('reconstruction_error(n_component=%d) : %s' % (n, lle.reconstruction_error_))  # 计算重构误差


# 降维后的样本分布图/k=1, 5出现了断路的现象
def plot_LocallyLinearEmbedding_k(*data):
    x, y = data
    Ks = [1, 5, 25, y.size - 1]
    fig = plt.figure()
    for i, k in enumerate(Ks):
        lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=k)  # n_neighbors 近邻参数k
        x_r = lle.fit_transform(x)
        ax = fig.add_subplot(2, 2, i + 1)
        colors = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
            , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label  # 这里的position类型为bool类型
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title('K=%d' % k)
    plt.suptitle('LocallyLinearEmbedding')
    plt.show()


# 最后给出将原始数据直接压缩至一维的情况
def plot_LocallyLinearEmbedding_k_d1(*data):
    x, y = data
    Ks = [1, 5, 25, y.size-1]
    fig = plt.figure()
    for i, k in enumerate(Ks):
        lle = manifold.LocallyLinearEmbedding(n_components=1, n_neighbors=k)  # n_neighbors 近邻参数k
        x_r = lle.fit_transform(x)
        ax = fig.add_subplot(2, 2, i + 1)
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0) \
            , (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label  # 这里的position类型为bool类型
            ax.scatter(x_r[position], np.zeros_like(x_r[position]), label="target=%d" % label, color=color)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="best")
        ax.set_title('K=%d' % k)
    plt.suptitle('LocallyLinearEmbedding')
    plt.show()