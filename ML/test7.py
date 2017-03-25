# svm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, cross_validation, svm


'''
    使用的数据集来自scikit-learn自带的一个糖尿病病人的数据集
    数据集有442个样本
    每个样本有10个特点
    每个特征都是浮点数，数据都是在-0.2~0.2之间
    样本的目标在整数25~346之间
'''

# 加载数据集的函数/用于支持向量机回归问题
def load_data_regression():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)


# 加载数据集的函数/用于支持向量机分类问题
# 数据集来自scikti-learn自带的xx花数据集
def load_data_classification():
    iris = datasets.load_iris()
    return cross_validation.train_test_split(iris.data, iris.target, test_size=0.25, random_state=0, stratify=iris.target)  # 分层采样


# 线性分类支持向量机的预测能力/实验结果表明线性分类支持向量机的预测性能相当好
def test_LinearSVC(*data):
    x_train, x_test, y_train, y_test = data
    cls = svm.LinearSVC()
    cls.fit(x_train, y_train)
    print("Coefficients:%s, intercept:%s" % (cls.coef_, cls.intercept_))
    print("Score: %.2f" % cls.score(x_test, y_test))


# 考察损失函数对原始算法的影响/实验表明，虽然支持向量机的损失函数不同，但是它们对于测试集的预测准确率都相同
def test_LinearSVC_loss(*data):
    x_train, x_test, y_train, y_test = data
    losses = ['hinge', 'squared_hinge']
    for loss in losses:
        cls = svm.LinearSVC(loss=loss)  # loss损失函数
        cls.fit(x_train, y_train)
        print("loss: %s" % loss)
        print("Coefficients:%s, intercept:%s" % (cls.coef_, cls.intercept_))
        print("Score: %.2f" % cls.score(x_test, y_test))


# 考察罚项不同的影响/实验表明，对于该数据集合，影响不大
def test_LinearSVC_L12(*data):
    x_train, x_test, y_train, y_test = data
    L12 = ['l1', 'l2']
    for p in L12:
        cls = svm.LinearSVC(penalty=p, dual=False)  # 罚项/是否为对偶/考虑到当p='l2'时候，dual=True情况不支持
        cls.fit(x_train, y_train)
        print("penalty: %s" % p)
        print("Coefficients:%s, intercept:%s" % (cls.coef_, cls.intercept_))
        print("Score: %.2f" % cls.score(x_test, y_test))


# 最后考察罚项系数C的影响，C衡量了误分类点的重要性，C越大则误分类点越重要/实验表明在c小的时候也就是误分类点重要性小的时候，误分类点确实多一点
def test_LinearSVC_C(*data):
    x_train, x_test, y_train, y_test = data
    Cs = np.logspace(-2, 1)
    train_scores = []
    test_scores = []
    for C in Cs:
        cls = svm.LinearSVC(C=C)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, train_scores, label='train_scores')
    ax.plot(Cs, test_scores, label='test_scores')
    ax.set_xlabel('Cs')
    ax.set_ylabel('Score')
    ax.set_title('LinearSVC')
    ax.set_xscale('log')
    ax.legend(loc='best')
    plt.show()


# 非线性分类SVM/训练的时间复杂度是采样点数量的平方
# 这里主要观察不同的核函数对预测性能的影响，首先观察最简单的线性核函数
# 实验结果表明支持线性核函数的非线性svm表现要比线性分类支持向量机LinearSVC的预测效果更佳，对测试集的预测全部正确
def test_SVC_linear(*data):
    x_train, x_test, y_train, y_test = data
    cls = svm.SVC(kernel='linear')
    cls.fit(x_train, y_train)
    print("Coefficients:%s, intercept:%s" % (cls.coef_, cls.intercept_))
    print("Score: %.2f" % cls.score(x_test, y_test))


# 考虑多项式核函数，degree/gamma/coef0均为服务于核函数的相关参数
# 实验结果表明在测试集上的预测性能随degree变化比较平稳，gamma影响不是很大，在r=0时性能最佳
def test_SVC_poly(*data):
    x_train, x_test, y_train, y_test = data
    fig = plt.figure()
    # 测试　degree
    degrees = range(1, 20)
    train_scores = []
    test_scores = []
    for degree in degrees:
        cls = svm.SVC(kernel='poly', degree=degree)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(degrees, train_scores, label='train_score')
    ax.plot(degrees, test_scores, label='test_score')
    ax.set_xlabel('degree')
    ax.set_ylabel('score')
    ax.set_title('SVC_Poly_Degree')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    # 测试gamma
    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        cls = svm.SVC(kernel='poly', gamma=gamma, degree=3)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gammas, train_scores, label='train_score')
    ax.plot(gammas, test_scores, label='test_score')
    ax.set_xlabel('gamma')
    ax.set_ylabel('score')
    ax.set_title('SVC_Poly_Gamma')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    # 测试coef0
    rs = range(0, 20)
    train_scores = []
    test_scores = []
    for r in rs:
        cls = svm.SVC(kernel='poly', degree=3, gamma=10, coef0=r)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(rs, train_scores, label='train_score')
    ax.plot(rs, test_scores, label='test_score')
    ax.set_xlabel('r')
    ax.set_ylabel('score')
    ax.set_title('SVC_Poly_r')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    plt.show()


# 考虑高斯核有gamma参数
def test_SVC_rbf(*data):
    x_train, x_test, y_train, y_test = data
    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    fig = plt.figure()
    for gamma in gammas:
        cls = svm.SVC(kernel='rbf', gamma=gamma)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(gammas, train_scores, label='train_score')
    ax.plot(gammas, test_scores, label='test_score')
    ax.set_xlabel('gamma')
    ax.set_ylabel('score')
    ax.set_title('SVC_Rbf_Gamma')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    plt.show()


# 最后考虑sigmoid核函数　gamma参数，coef0参数
def test_SVC_sigmoid(*data):
    x_train, x_test, y_train, y_test = data
    fig = plt.figure()

    # 测试gamma
    gammas = np.logspace(-2, 1)
    train_scores = []
    test_scores = []

    for gamma in gammas:
        cls = svm.SVC(kernel='sigmoid', gamma=gamma, coef0=0)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(gammas, train_scores, label='train_score')
    ax.plot(gammas, test_scores, label='test_score')
    ax.set_xlabel('gamma')
    ax.set_ylabel('score')
    ax.set_xscale('log')
    ax.set_title('SVC_Sigmoid_Gamma')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度

    # 测试r
    rs = np.linspace(0, 5)
    train_scores = []
    test_scores = []

    for r in rs:
        cls = svm.SVC(kernel='sigmoid', gamma=0.01, coef0=r)
        cls.fit(x_train, y_train)
        train_scores.append(cls.score(x_train, y_train))
        test_scores.append(cls.score(x_test, y_test))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(gammas, train_scores, label='train_score')
    ax.plot(gammas, test_scores, label='test_score')
    ax.set_xlabel('r')
    ax.set_ylabel('score')
    ax.set_title('SVC_Sigmoid_r')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    fig.suptitle('SIGMOID')
    plt.show()


# 线性回归SVR/实验表明线性支持向量机预测性能较差
def test_LinearSVR(*data):
    x_train, x_test, y_train, y_test = data
    regr = svm.LinearSVR()
    regr.fit(x_train, y_train)
    print("Coefficients:%s, intercept:%s" % (regr.coef_, regr.intercept_))
    print("Score: %.2f" % regr.score(x_test, y_test))


# 考虑损失函数对于预测性能的影响/实验表明loss=squared_epsilon_insensitive时，预测性能更好
def test_LinearSVR_loss(*data):
    x_train, x_test, y_train, y_test = data
    losses = ['epsilon_insensitive', 'squared_epsilon_insensitive']
    for loss in losses:
        regr = svm.LinearSVR(loss=loss)
        regr.fit(x_train, y_train)
        print("loss: %s" % loss)
        print("Coefficients:%s, intercept:%s" % (regr.coef_, regr.intercept_))
        print("Score: %.2f" % regr.score(x_test, y_test))


# epsion对于预测性能的影响，该参数代表只要预测值落在标准值附近epsilon宽的区间内，都标记为预测正确
def test_LinearSVR_epsilon(*data):
    x_train, x_test, y_train, y_test = data
    epsilons = np.logspace(-2, 2)
    fig = plt.figure()
    train_scores = []
    test_scores = []

    for epsilon in epsilons:
        regr = svm.LinearSVR(epsilon=epsilon, loss='squared_epsilon_insensitive')
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epsilons, train_scores, label='train_score')
    ax.plot(epsilons, test_scores, label='test_score')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('score')
    ax.set_xscale('log')
    ax.set_title('LinearSVR_epsilon')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    plt.show()


# 罚项系数C
def test_LinearSVR_C(*data):
    x_train, x_test, y_train, y_test = data
    Cs = np.logspace(-1, 2)
    train_scores = []
    test_scores = []
    for C in Cs:
        regr = svm.LinearSVR(C=C, epsilon=0.1, loss='squared_epsilon_insensitive')
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, train_scores, label='train_scores')
    ax.plot(Cs, test_scores, label='test_scores')
    ax.set_xlabel('Cs')
    ax.set_ylabel('Score')
    ax.set_title('LinearSVR')
    ax.set_ylim(0, 1)
    ax.set_xscale('log')
    ax.legend(loc='best')
    plt.show()


# 非线性回归SVR/线性核
def test_SVR_linear(*data):
    x_train, x_test, y_train, y_test = data
    regr = svm.SVR(kernel='linear')
    regr.fit(x_train, y_train)
    print("Coefficients:%s, intercept:%s" % (regr.coef_, regr.intercept_))
    print("Score: %.2f" % regr.score(x_test, y_test))

# 非线性回归SVR/多项式核函数
def test_SVR_poly(*data):
    x_train, x_test, y_train, y_test = data
    fig = plt.figure()
    # 测试　degree
    degrees = range(1, 20)
    train_scores = []
    test_scores = []
    for degree in degrees:
        regr = svm.SVR(kernel='poly', degree=degree, coef0=1)
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(degrees, train_scores, label='train_score')
    ax.plot(degrees, test_scores, label='test_score')
    ax.set_xlabel('degree')
    ax.set_ylabel('score')
    ax.set_title('SVC_Poly_Degree')
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    # 测试gamma
    gammas = range(1, 40)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        regr = svm.SVR(kernel='poly', gamma=gamma, degree=3, coef0=1)
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gammas, train_scores, label='train_score')
    ax.plot(gammas, test_scores, label='test_score')
    ax.set_xlabel('gamma')
    ax.set_ylabel('score')
    ax.set_title('SVC_Poly_Gamma')
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    # 测试coef0
    rs = range(0, 20)
    train_scores = []
    test_scores = []
    for r in rs:
        regr = svm.SVR(kernel='poly', degree=3, gamma=20, coef0=r)
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(rs, train_scores, label='train_score')
    ax.plot(rs, test_scores, label='test_score')
    ax.set_xlabel('r')
    ax.set_ylabel('score')
    ax.set_title('SVC_Poly_r')
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    plt.show()


# 考虑高斯核有gamma参数
def test_SVR_rbf(*data):
    x_train, x_test, y_train, y_test = data
    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    fig = plt.figure()
    for gamma in gammas:
        regr = svm.SVR(kernel='rbf', gamma=gamma)
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(gammas, train_scores, label='train_score')
    ax.plot(gammas, test_scores, label='test_score')
    ax.set_xlabel('gamma')
    ax.set_ylabel('score')
    ax.set_title('SVC_Rbf_Gamma')
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    plt.show()


# 最后考虑sigmoid核函数　gamma参数，coef0参数
def test_SVR_sigmoid(*data):
    x_train, x_test, y_train, y_test = data
    fig = plt.figure()

    # 测试gamma
    gammas = np.logspace(-1, 3)
    train_scores = []
    test_scores = []

    for gamma in gammas:
        regr = svm.SVR(kernel='sigmoid', gamma=gamma, coef0=0.01)
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(gammas, train_scores, label='train_score')
    ax.plot(gammas, test_scores, label='test_score')
    ax.set_xlabel('gamma')
    ax.set_ylabel('score')
    ax.set_xscale('log')
    ax.set_title('SVC_Sigmoid_Gamma')
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度

    # 测试r
    rs = np.linspace(0, 5)
    train_scores = []
    test_scores = []

    for r in rs:
        regr = svm.SVR(kernel='sigmoid', gamma=10, coef0=r)
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(gammas, train_scores, label='train_score')
    ax.plot(gammas, test_scores, label='test_score')
    ax.set_xlabel('r')
    ax.set_ylabel('score')
    ax.set_title('SVC_Sigmoid_r')
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best', framealpha=0.5)  # 第二个参数是控制frame的透明度
    fig.suptitle('SIGMOID')
    plt.show()