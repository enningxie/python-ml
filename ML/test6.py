# 聚类与EM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import  make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture


# 处理数据
def create_data(center, num=100, std=0.7):
    '''
        centers:聚类的中心点组成的数组；
        num 样本数
        std 每个簇中的样本的标准差
        返回值：一个元组，第一个元素为样本点，第二个元素为样本点的真实簇分类标记
    '''
    x, labels_true = make_blobs(n_samples=num, centers=center, cluster_std=std)  # 该函数产生的是分隔的高斯分布的聚类簇
    return x, labels_true


#　给出生成样本点的图像/结果产生了四个簇，为了考察聚类的性能，将三个簇交织在一起，另一个簇比较远
def plot_data(*data):
    x, labels_true = data
    labels = np.unique(labels_true)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = 'rgbyckm'
    for i, label in enumerate(labels):
        position = labels_true == label
        color = colors[i % len(colors)]
        ax.scatter(x[position, 0], x[position, 1], label="cluster %d" % label, color=color)

    ax.legend(loc='best', framealpha=0.5)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('y[1]')
    ax.set_title("data")
    plt.show()


# KMeans/ARI指标越大越好
def test_KMeans(*data):
    x, label_true = data
    clst = cluster.KMeans()
    clst.fit(x)
    predicted_labels = clst.predict(x)  # 预测样本所属的簇
    print("ARI:%s" % adjusted_rand_score(label_true, predicted_labels))
    print("Sum center distance %s " % clst.inertia_)  # 每个样本距离他们簇中心的距离之和
    # print(np.unique(clst.labels_))


# 考察簇的数量的影响
def test_KMeans_nclusters(*data):
    x, labels_true = data
    nums = range(1, 50)
    ARIs = []
    Distances = []
    for num in nums:
        clst = cluster.KMeans(n_clusters=num)
        clst.fit(x)
        predicted_labels = clst.predict(x)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances.append(clst.inertia_)
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, ARIs, marker='+')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, Distances, marker='o')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("Distances")
    fig.suptitle("KMeans")
    plt.show()


# 考察kmeans算法运行的次数和选择初始中心向量策略的影响/实验表明这两项对算法整体影响并不是很大。
def test_KMeans_n_init(*data):
    x, y = data
    nums = range(1, 50)
    # 绘图
    fig = plt.figure()
    ARIs = []
    Distances = []
    ARIs_r = []
    Distances_r = []
    for num in nums:
        clst = cluster.KMeans(n_init=num, init='k-means++')  # n_init指定了算法的运行次数／init指定初始均值向量的策略
        clst.fit(x)
        predicted_labels = clst.predict(x)
        ARIs.append(adjusted_rand_score(y, predicted_labels))
        Distances.append(adjusted_rand_score(y, predicted_labels))
        clst2 = cluster.KMeans(n_init=num, init='random')
        clst2.fit(x)
        predicted_labels2 = clst2.predict(x)
        ARIs_r.append(adjusted_rand_score(y, predicted_labels2))
        Distances_r.append(adjusted_rand_score(y, predicted_labels2))
# 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, ARIs, marker='+', label='k-means++')
    ax.plot(nums, ARIs_r, marker='+', label='random')
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, Distances, marker='o', label='k-means++')
    ax.plot(nums, Distances_r, marker='o', label='random')
    ax.set_xlabel("n_init")
    ax.set_ylabel("Distances")
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    fig.suptitle("KMeans")
    plt.show()


# 密度聚类／DBSCAN根据密度将原始数据集分为core_sample_indices个簇
def test_DBSCAN(*data):
    x, y = data
    clst = cluster.DBSCAN()
    predicted_labels = clst.fit_predict(x)  # 训练模型并预测每个样本所属的簇标记
    print("ARI: %s" % adjusted_rand_score(y, predicted_labels))
    print("Core sample num: %d" % len(clst.core_sample_indices_))  # core_sample_indices_核心样本所在原始样本的位置


# 考察epsilon参数的影响/该参数是用来确定领域的大小
def test_DBSCAN_epsilon(*data):
    x, y = data
    epsilons = np.logspace(-1, 1.5)  # 默认num=50
    ARIs = []
    Core_nums = []
    for epsilon in epsilons:
        clst = cluster.DBSCAN(eps=epsilon)
        predicted_labels = clst.fit_predict(x)
        ARIs.append(adjusted_rand_score(y, predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(epsilons, ARIs, marker='+')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylim(0, 1)
    ax.set_ylabel('ARI')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(epsilons, Core_nums, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Core_nums')
    fig.suptitle('DBSCAN')
    plt.show()


# 考察MinPts参数的影响/用于判断核心对象
def test_DBSCAN_min_samples(*data):
    x, y = data
    min_samples = range(1, 100)
    ARIs = []
    Core_nums = []
    for num in min_samples:
        clst = cluster.DBSCAN(min_samples=num)
        predicted_labels = clst.fit_predict(x)
        ARIs.append(adjusted_rand_score(y, predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(min_samples, ARIs, marker='+')
    ax.set_xlabel("min_sample")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(min_samples, Core_nums, marker='o')
    ax.set_xlabel("min_sample")
    ax.set_ylabel("Core_num")
    fig.suptitle('DBSCAN')
    plt.show()


# 层次聚类
def test_AgglomerativeClustering(*data):
    x, y = data
    clst = cluster.AgglomerativeClustering()
    predicted_labels = clst.fit_predict(x)
    print("ARI:%s" % adjusted_rand_score(y, predicted_labels))


# 考察簇的数量对于聚类效果的影响
def test_AgglomerativeClustering_nclusters(*data):
    x, y = data
    nums = range(1, 50)
    ARIs = []
    for num in nums:
        clst = cluster.AgglomerativeClustering(n_clusters=num)
        predicted_labels = clst.fit_predict(x)
        ARIs.append(adjusted_rand_score(y, predicted_labels))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nums, ARIs, marker='o')
    ax.set_xlabel('n_cluster')
    ax.set_ylim(0, 1)
    ax.set_ylabel('ARI')
    ax.set_title('AgglomerativeClustering')
    plt.show()


# 考察链接方式的影响/'ward'方式最好
def test_AgglomerativeClustering_linkage(*data):
    x, y = data
    nums = range(1, 50)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    linkages = ['ward', 'complete', 'average']
    marker = '+o*'
    for i, linkage in enumerate(linkages):
        ARIs = []
        for num in nums:
            clst = cluster.AgglomerativeClustering(n_clusters=num, linkage=linkage)
            predicted_labels = clst.fit_predict(x)
            ARIs.append(adjusted_rand_score(y, predicted_labels))
        ax.plot(nums, ARIs, marker=marker[i], label=linkage)
    ax.set_xlabel('n_clusters')
    ax.set_ylim(0, 1)
    ax.set_ylabel('linkage')
    ax.set_title("AgglomerativeClustering")
    plt.show()


# 混合高斯模型/默认的GMM只有一个簇
def test_GMM(*data):
    x, y = data
    clst = mixture.GMM()
    predicted_labels = clst.fit_predict(x)
    print("ARI:%s" % adjusted_rand_score(y, predicted_labels))


# 考察n_components／指定分模型的数量参数
def test_GMM_n_components(*data):
    x, y = data
    nums = range(1, 50)
    ARIs = []
    for num in nums:
        clst = mixture.GMM(n_components=num)
        predicted_labels = clst.fit_predict(x)
        ARIs.append(adjusted_rand_score(y, predicted_labels))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nums, ARIs)
    ax.set_xlabel("n_component")
    ax.set_ylabel("ARI")
    ax.set_title("GMM")
    plt.show()


# 考察协方差类型的影响/协方差矩阵的类型对整体影响不大
def test_GMM_cov_type(*data):
    x, y = data
    nums = range(1, 50)
    cov_types = ['spherical', 'tied', 'diag', 'full']
    marker = '+o*s'
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, cov_type in enumerate(cov_types):
        ARIs = []
        for num in nums:
            clst = mixture.GMM(n_components=num, covariance_type=cov_type)
            predited_labels = clst.fit_predict(x)
            ARIs.append(adjusted_rand_score(y, predited_labels))
        ax.plot(nums, ARIs, marker=marker[i], label='%s' % cov_type)
    ax.set_xlabel("n_component")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    ax.set_title("GMM")
    plt.show()
