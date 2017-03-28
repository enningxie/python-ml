# 数据的预处理


# 二元化
def test_binarizer():
    x = [
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1]
    ]
    from sklearn.preprocessing import Binarizer
    print("before transform:", x)
    binarizer = Binarizer(threshold=2.5)  # threshold参数指定了属性的阈值
    print("after transform:", binarizer.transform(x))


# 独热码
def test_oneHotEncoder():
    x = [
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1]
    ]
    from sklearn.preprocessing import OneHotEncoder
    print("before transform:", x)
    encoder = OneHotEncoder(sparse=False)  # sparse指定是否稀疏
    encoder.fit(x)
    print("active_features_:", encoder.active_features_)
    print("feature_indices_:", encoder.feature_indices_)
    print("n_values_:", encoder.n_values_)  # 存放了每个属性取值的种类
    print("after transform:", encoder.transform([[1, 2, 3, 4, 5]]))


# 标准化
def test_min_max_scaler():
    from sklearn.preprocessing import MinMaxScaler
    x = [
        [1, 5, 1, 2, 10],
        [2, 6, 3, 2, 7],
        [3, 7, 5, 6, 4],
        [4, 8, 7, 8, 1]
    ]
    print("before transform:", x)
    scaler = MinMaxScaler(feature_range=(0, 2))  # 指定了预期变换后属性的取值范围
    scaler.fit(x)
    print("min_ is:", scaler.min_)
    print("scale_is:", scaler.scale_)
    print("data_max_ is:", scaler.data_max_)
    print("data_min_ is:", scaler.data_min_)
    print("data_range_ is:", scaler.data_range_)
    print("after transform:", scaler.transform(x))


# 标准化后每个属性的绝对值都在[0, 1]中
def test_max_abs_scaler():
    from sklearn.preprocessing import MaxAbsScaler
    x = [
        [1, 5, 1, 2, 10],
        [2, 6, 3, 2, 7],
        [3, 7, 5, 6, 4],
        [4, 8, 7, 8, 1]
    ]
    print("before transform:", x)
    scaler = MaxAbsScaler()
    scaler.fit(x)
    print("scale_ is:", scaler.scale_)
    print("max_abs_ is:", scaler.max_abs_)
    print("after transform:", scaler.transform(x))


# standardScaler
def test_standar_scaler():
    from sklearn.preprocessing import StandardScaler
    x = [
        [1, 5, 1, 2, 10],
        [2, 6, 3, 2, 7],
        [3, 7, 5, 6, 4],
        [4, 8, 7, 8, 1]
    ]
    print("before transform:", x)
    scaler = StandardScaler()
    scaler.fit(x)
    print("scale_ is:", scaler.scale_)
    print("mean_ is:", scaler.mean_)
    print("var_ is:", scaler.var_)
    print("after transform:", scaler.transform(x))


# 正则化
def test_normalizer():
    from sklearn.preprocessing import Normalizer
    x = [
        [1., 2, 3, 4, 5],
        [5., 4, 3, 2, 1],
        [1., 3, 5, 2, 4],
        [2., 4, 1, 3, 5]
    ]
    print("before transform:", x)
    normalizer = Normalizer(norm='l2')  # 指定正则化方法
    print("after transform:", normalizer.transform(x))


# 过滤式特征选取
def test_varianceThreshold():
    from sklearn.feature_selection import VarianceThreshold
    x = [
        [100, 1, 2, 3],
        [100, 4, 5, 6],
        [100, 7, 8, 9],
        [101, 11, 12, 13]
    ]
    selector = VarianceThreshold(1)
    selector.fit(x)
    print("Variances is %s" % selector.variances_)
    print("After transform is %s" % selector.transform(x))
    print("The support is %s" % selector.get_support(True))
    print("After reverse transform is %s" % selector.inverse_transform(selector.transform(x)))


# 单变量特征选取
# 单变量特征选取通过计算每个特征的某个统计指标，然后根据该指标来选取特征
def test_select():
    from sklearn.feature_selection import SelectKBest, f_classif
    x = [
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1]
    ]
    y = [0, 1, 0, 1]
    print("before transform:", x)
    selector = SelectKBest(score_func=f_classif, k=3)
    selector.fit(x, y)
    print("score:", selector.scores_)
    print("pvalues:", selector.pvalues_)
    print("selected index:", selector.get_support(True))
    print("after transform:", selector.transform(x))


# 包裹式特征选取
# RFE通过外部提供的一个学习器来选择特征/要求学习器学习的是特征的权重
def test_select2():
    from sklearn.feature_selection import RFE
    from sklearn.datasets import load_iris
    from sklearn.svm import LinearSVC
    iris = load_iris()
    x = iris.data
    y = iris.target
    estimator = LinearSVC()
    selector = RFE(estimator=estimator, n_features_to_select=2)
    selector.fit(x, y)
    print("N_features %s" % selector.n_features_)
    print("Support is %s" % selector.support_)
    print("Ranking %s" % selector.ranking_)


# 特征选取对于预测性能的提升没有必然的联系
def test_select3():
    from sklearn.feature_selection import RFE
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    ## 加载数据
    iris = load_iris()
    x, y = iris.data, iris.target
    ## 特征提取
    estimator = LinearSVC()
    selector = RFE(estimator=estimator, n_features_to_select=2)
    x_t = selector.fit_transform(x, y)
    ## 切分测试集与验证集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_t, y, test_size=0.25, random_state=0, stratify=y)
    ## 测试与验证
    clf = LinearSVC()
    clf_t = LinearSVC()
    clf.fit(x, y)
    clf_t.fit(x_t, y)
    print("Original DataSet:", clf.score(x_test, y_test))
    print("Selected DataSet:", clf_t.score(x_test_t, y_test))


# RFEC/它是RFEC类，它是RFE的一个变体，它执行一个交叉验证来寻找最优的剩余特征数量，因此不用保证留多少个特征
def test_select4():
    import numpy as np
    from sklearn.feature_selection import RFECV
    from sklearn.svm import LinearSVC
    from sklearn.datasets import load_iris
    iris = load_iris()
    x = iris.data
    y = iris.target
    estimator = LinearSVC()
    selector = RFECV(estimator=estimator, cv=3)
    selector.fit(x, y)
    print("N_features %s" % selector.n_features_)  # 选出的特征数
    print("Support is %s" % selector.support_)
    print("Ranking %s" % selector.ranking_)  # rank
    print("Grid Scores %s" % selector.grid_scores_)  # 给出单个特征上交叉验证得到的最佳预测准确率


# 嵌入式特征选取
def test_selectFromModel():
    import numpy as np
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC
    from sklearn.datasets import load_digits
    digits = load_digits()
    x = digits.data
    y = digits.target
    estimator = LinearSVC(penalty='l1', dual=False)
    selector = SelectFromModel(estimator=estimator, threshold='mean')
    selector.fit(x, y)
    selector.transform(x)
    print("Threshold %s" % selector.threshold_)
    print("Support is %s" % selector.get_support(indices=True))


# 稀疏性的test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits,load_diabetes
from sklearn.linear_model import Lasso


def test_Lasso():
    diabetes = load_diabetes()
    x = diabetes.data
    y = diabetes.target
    alphas = np.logspace(-2, 2)
    zeros = []
    for alpha in alphas:
        regr = Lasso(alpha=alpha)
        regr.fit(x, y)
        ### 计算0的个数
        num = 0
        for ele in regr.coef_:
            if abs(ele) < 1e-5:
                num += 1
        zeros.append(num)
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, zeros)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("num")
    ax.set_xscale("log")
    ax.set_ylim(0, x.shape[1] + 1)
    plt.show()


def test_LinearSVC():
    digits = load_digits()
    x = digits.data
    y = digits.target
    Cs = np.logspace(-2, 2)
    zeros = []
    for C in Cs:
        clf = LinearSVC(C=C, penalty='l1', dual=False)
        clf.fit(x, y)
        num = 0
        for row in clf.coef_:
            for ele in row:
                if abs(ele) < 1e-5:
                    num += 1
        zeros.append(num)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, zeros)
    ax.set_xlabel("C")
    ax.set_ylabel("num")
    ax.set_xscale("log")
    plt.show()


# 学习器流水线
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_Pipeline(*data):
    d = load_digits()
    x_trian, x_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.25, random_state=0, stratify=d.target)
    steps = [("Linear_SVM", LinearSVC(C=1, penalty='l1', dual=False)), ("LogisticRegression", LogisticRegression(C=1))]
    pipeline = Pipeline(steps)
    pipeline.fit(x_trian, y_train)
    print("Named steps:", pipeline.named_steps)
    print("Pipeline Score:", pipeline.score(x_test, y_test))


# 字典学习
def test_DictionaryLearning():
    from sklearn.decomposition import DictionaryLearning
    x = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6],
        [5, 4, 3, 2, 1]
    ]
    print("before transform:", x)
    dct = DictionaryLearning(n_components=3)
    dct.fit(x)
    print("components is :", dct.components_)
    print("after transform:", dct.transform(x))