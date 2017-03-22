
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split


# 加载数据集（糖尿病数据集）
def load_data():
    diabetes = datasets.load_diabetes()
    return train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)


# 加载数据集（iris数据集）
def load_data2():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


# 线性回归 Coefficients系数/intercept截距
def test_linearRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%.2f' % (regr.coef_, regr.intercept_))
    print('Residual sum of squares: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


# 岭回归
def test_Ridge(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%.2f' % (regr.coef_, regr.intercept_))
    print('Residual sum of squares: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


# 岭回归测试，检验不同的alpha对于预测性能的影响。
def test_Ridge_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Ridge")
    plt.show()


# lasso回归
def test_Lasso(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%.2f' % (regr.coef_, regr.intercept_))
    print('Residual sum of squares: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


# lasso回归不同alpha测试
def test_Lasso_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Lasso")
    plt.show()


# ElasticNet 回归
def test_ElasticNet(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.ElasticNet()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%.2f' % (regr.coef_, regr.intercept_))
    print('Residual sum of squares: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


#  检测不同的超参数对上述回归的影响
def test_ElasticNet_alpha_rho(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2, 2)
    rhos = np.linspace(0.01, 1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho)
            regr.fit(X_train, y_train)
            scores.append(regr.score(X_test, y_test))
    # 绘图
    alphas, rhos = np.meshgrid(alphas, rhos)    # xz
    scores = np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel(r"score")
    ax.set_title("ElasticNet")
    plt.show()


# 逻辑回归
def test_LogisticRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%s' % (regr.coef_, regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))
    # print(regr.predict_proba(X_test).shape)


# logistic回归中，基于不同的多分类策略问题（multi_class）
def test_LogisticRegression_multinomial(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%s' % (regr.coef_, regr.intercept_))
    print('Score: %.2f' % regr.score(X_test, y_test))


# 模型正则化项对模型的影响
def test_LogisticRegression_C(*data):
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-2, 4, num=100)
    scores=[]
    for C in Cs:
        regr = linear_model.LogisticRegression(C=C)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    #绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, scores)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LogsticRegression")
    plt.show()


# 线性判别分析
def test_LinearDiscriminantAnalysis(*data):
    X_train, X_test, y_train, y_test = data
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%s' % (lda.coef_, lda.intercept_))
    print('Score: %.2f' % lda.score(X_test, y_test))


# 图解LDA
def plot_LDA(converted_X, y):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = 'rgb'
    markers = 'o*s'
    for target, color, marker in zip([0,1,2], colors, markers):
        pos = (y == target).ravel()
        X = converted_X[pos, :]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=marker, label="Label %d" % target)
        ax.legend(loc="best")
        fig.suptitle("Iris After LDA")
        plt.show()


# 不同solver对LDA的影响
def test_linearDiscriminantAnalysis_solver(*data):
    X_train, X_test, y_train, y_test = data
    solvers = ['svd', 'lsqr', 'eigen']
    for solver in solvers:
        if solver == 'svd':
            lda = discriminant_analysis.LinearDiscriminantAnalysis(solver=solver)
        else:
            lda = discriminant_analysis.LinearDiscriminantAnalysis(solver=solver, shrinkage=None)
        lda.fit(X_train, y_train)
        print('Score at solver=%s: %.2f' % (solver, lda.score(X_test, y_test)))


# 在solver=lsqr中引入抖动。相当于引入了正则化项。
def test_LinearDiscriminantAnalysis_shrinkage(*data):
    X_train, X_test, y_train, y_test = data
    shrinkages = np.linspace(0.0, 1.0, num=20)
    scores = []
    for shrinkage in shrinkages:
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
        lda.fit(X_train, y_train)
        scores.append(lda.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(shrinkages, scores)
    ax.set_xlabel(r"shrinkage")
    ax.set_ylabel(r"score")
    ax.set_ylim(0, 1.05)
    ax.set_title("LinearDiscriminantAnalysis")
    plt.show()