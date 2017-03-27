# 集成学习
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble  # 集成
from sklearn import model_selection


# 加载数据集/糖尿病人的数据集用于回归
def load_data_regression():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)


# 加载数据集/手写数字识别的数据集用于分类
def load_data_classification():
    digits = datasets.load_digits()
    return model_selection.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0, stratify=digits.target)


# AdaBoostClassifier分类器
# 实验表明随着迭代的增加，集成分类器的误差都在下降，当分类器增加到一定的程度，误差趋于稳定
# 集成学习能够很好地抵抗过拟合
def test_AdaBoostClassifier(*data):
    x_train, x_test, y_train, y_test = data
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(x_train, y_train)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    estimators_num = len(clf.estimators_)  # clf.estimators_:所有训练过的基础分类器
    x = range(1, estimators_num+1)
    ax.plot(list(x), list(clf.staged_score(x_train, y_train)), label="training score")  # staged_score:每一轮迭代结束时尚未完成的集成分类器的预测准确率
    ax.plot(list(x), list(clf.staged_score(x_test, y_test)), label="testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()


# 考察不同类型的个体分类器的影响，并给出测试函数/个体分类器为强分类器的效果要好
def test_AdaBoostClassifier_base_classifier(*data):
    from sklearn.naive_bayes import GaussianNB
    x_train, x_test, y_train, y_test = data
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ##　默认个体分类器
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(x_train, y_train)
    estimators_num = len(clf.estimators_)
    x = range(1, estimators_num+1)
    ax.plot(list(x), list(clf.staged_score(x_train, y_train)), label="training score")
    ax.plot(list(x), list(clf.staged_score(x_test, y_test)), label="testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("AdaBoostClassifier with Decision Tree")
    ## GaussianNB
    ax = fig.add_subplot(2, 1, 2)
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1, base_estimator=GaussianNB())
    clf.fit(x_train, y_train)
    estimators_num = len(clf.estimators_)
    x = range(1, estimators_num + 1)
    ax.plot(list(x), list(clf.staged_score(x_train, y_train)), label="training score")
    ax.plot(list(x), list(clf.staged_score(x_test, y_test)), label="testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("AdaBoostClassifier with GaussianNB")
    plt.show()


# 考察学习率的影响
def test_AdaBoostClassifier_learning_rate(*data):
    x_train, x_test, y_train, y_test = data
    learning_rates = np.linspace(0.01, 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    train_scores = []
    test_scores = []
    for learning_rate in learning_rates:
        clf = ensemble.AdaBoostClassifier(learning_rate=learning_rate, n_estimators=500)
        clf.fit(x_train, y_train)
        train_scores.append(clf.score(x_train, y_train))
        test_scores.append(clf.score(x_test, y_test))
    ax.plot(learning_rates, train_scores, label="train_score")
    ax.plot(learning_rates, test_scores, label="test_score")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("AdaBoost_learning_rate")
    ax.legend(loc='best')
    plt.show()


# 考察分类算法的影响
def test_AdaBoostClassifier_algorithm(*data):
    x_train, x_test, y_train, y_test = data
    algorithms = ['SAMME.R', 'SAMME']
    fig = plt.figure()
    learning_rates = [0.05, 0.1, 0.5, 0.9]
    for i, learning_rate in enumerate(learning_rates):
        ax = fig.add_subplot(2, 2, i+1)
        for j, algorithm in enumerate(algorithms):
            clf = ensemble.AdaBoostClassifier(learning_rate=learning_rate, algorithm=algorithm)
            clf.fit(x_train, y_train)
            estimators_num = len(clf.estimators_)
            x = range(1, estimators_num+1)
            ax.plot(list(x), list(clf.staged_score(x_train, y_train)), label="train_alg:%s" % algorithms[j])
            ax.plot(list(x), list(clf.staged_score(x_test, y_test)), label="test_alg:%s" % algorithms[j])
        ax.set_xlabel("estimator_num")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        ax.set_title("learning_rate:%f" % learning_rates[i])
    fig.suptitle("AdaboostClassifier")
    plt.show()


# AdaBoostRegression回归器
def test_AdaBoostRegressor(*data):
    x_train, x_test, y_train, y_test = data
    regr = ensemble.AdaBoostRegressor()
    regr.fit(x_train, y_train)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    estimators_num = len(regr.estimators_)
    x = range(1, estimators_num+1)
    ax.plot(list(x), list(regr.staged_score(x_train, y_train)), label="train_score")
    ax.plot(list(x), list(regr.staged_score(x_test, y_test)), label="test_score")
    ax.set_xlabel("estimators_num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax.set_title("AdaBoost_regression")
    plt.show()


# 考虑不同个体分类器的影响
def test_AdaBoostRegressor_base_regr(*data):
    from sklearn.svm import LinearSVR
    x_train, x_test, y_train, y_test = data
    fig = plt.figure()
    regrs = [ensemble.AdaBoostRegressor(), ensemble.AdaBoostRegressor(base_estimator=LinearSVR(epsilon=0.01, C=100))]
    labels = ["Decision Tree", "Linear SVM"]
    for i, regr in enumerate(regrs):
        regr.fit(x_train, y_train)
        ax = fig.add_subplot(1, 1, 1)
        estimators_num = len(regr.estimators_)
        x = range(1, estimators_num+1)
        ax.plot(list(x), list(regr.staged_score(x_train, y_train)), label="%s:train_score" % labels[i])
        ax.plot(list(x), list(regr.staged_score(x_test, y_test)), label="%s:test_score" % labels[i])
    ax.set_xlabel("estimators_num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax.set_title("AdaBoost_regression")
    plt.show()


# 考虑学习率的影响
def test_AdaBoostRegressor_learning_rate(*data):
    x_train, x_test, y_train, y_test = data
    learning_rates = np.linspace(0.01, 1)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for i, learning_rate in enumerate(learning_rates):
        regr = ensemble.AdaBoostRegressor(learning_rate=learning_rate)
        regr.fit(x_train, y_train)
        train_scores.append(regr.score(x_train, y_train))
        test_scores.append(regr.score(x_test, y_test))
    ax.plot(learning_rates, train_scores, label="train_score")
    ax.plot(learning_rates, test_scores, label="test_score")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("score")
    ax.set_title("AdaBoostRegression")
    ax.legend(loc='best')
    ax.set_ylim(0, 1)
    plt.show()


# 考察loos函数的影响
def test_AdaBoostRegressor_loss(*data):
    x_train, x_test, y_train, y_test = data
    losses = ['linear', 'square', 'exponential']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, loss in enumerate(losses):
        regr = ensemble.AdaBoostRegressor(loss=loss, n_estimators=30)
        regr.fit(x_train, y_train)
        estimators_num = len(regr.estimators_)
        x = range(1, estimators_num+1)
        ax.plot(list(x), list(regr.staged_score(x_test, y_test)), label='%s:test_score' % loss)
    ax.set_xlabel("estimator_num")
    ax.set_ylabel("score")
    ax.set_title("AdaBoostRegression")
    ax.legend(loc='best')
    plt.show()


# Gradient Tree Boosting/梯度提升决策树
def test_GradientBoostingClassifier(*data):
    x_train, x_test, y_train, y_test = data
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(x_train, y_train)
    print("train_socre:", clf.score(x_train, y_train))
    print("test_socre:", clf.score(x_test, y_test))


# 考虑个体决策树的数量对于上述算法性能的影响
def test_GradientBoostingClassifier_num(*data):
    x_train, x_test, y_train, y_test = data
    nums = np.arange(1, 100, 2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    test_scores = []
    for num in nums:
        clf = ensemble.GradientBoostingClassifier(n_estimators=num)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_test, y_test))
    ax.plot(nums, test_scores, label="test_score")
    ax.set_xlabel("n_estimator")
    ax.set_ylabel("score")
    ax.legend(loc='best')
    plt.show()


# 考虑个体决策树的最大树深对于算法的性能影响
def test_GradientBoostingClassifier_maxdepth(*data):
    x_train, x_test, y_train, y_test = data
    maxdepths = np.arange(1, 20)
    fig = plt.figure()
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for maxdepth in maxdepths:
        clf = ensemble.GradientBoostingClassifier(max_depth=maxdepth, max_leaf_nodes=None)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_train, y_train))
    ax.plot(maxdepths, test_scores, label="test_score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.legend(loc='best')
    plt.show()


# 学习率的影响
def test_GradientBoostingClassifier_learning(*data):
    x_train, x_test, y_train, y_test = data
    learning_rates = np.linspace(0.01, 1.0)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for learning_rate in learning_rates:
        clf = ensemble.GradientBoostingClassifier(learning_rate=learning_rate)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_train, y_train))
        train_scores.append(clf.score(x_test, y_test))
    ax.plot(learning_rates, test_scores, label="test_score")
    ax.plot(learning_rates, train_scores, label="train_score")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    plt.show()


# 考虑subsample的影响
def test_GradientBoostingClassifier_subsample(*data):
    x_train, x_test, y_train, y_test = data
    subsamples = np.linspace(0.01, 1.0)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for subsample in subsamples:
        clf = ensemble.GradientBoostingClassifier(subsample=subsample)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_train, y_train))
        train_scores.append(clf.score(x_test, y_test))
    ax.plot(subsamples, test_scores, label="test_score")
    ax.plot(subsamples, train_scores, label="train_score")
    ax.set_xlabel("subsample")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    plt.show()


# 考虑max_features的影响
def test_GradientBoostingClassifier_max_features(*data):
    x_train, x_test, y_train, y_test = data
    max_features = np.linspace(0.01, 1.0)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for max_feature in max_features:
        clf = ensemble.GradientBoostingClassifier(max_features=max_feature)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_train, y_train))
        train_scores.append(clf.score(x_test, y_test))
    ax.plot(max_features, test_scores, label="test_score")
    ax.plot(max_features, train_scores, label="train_score")
    ax.set_xlabel("max_feature")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    plt.show()


# 回归梯度提升树
def test_GradientBoostingRegressor(*data):
    x_train, x_test, y_train, y_test = data
    regr = ensemble.GradientBoostingRegressor()
    regr.fit(x_train, y_train)
    print("train_score:", regr.score(x_train, y_train))
    print("test_score:", regr.score(x_test, y_test))


# 考虑到个体回归树的数量的影响
def test_GradientBoostingRegressor_num(*data):
    x_train, x_test, y_train, y_test = data
    nums = np.arange(1, 200, 2)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for num in nums:
        regr = ensemble.GradientBoostingRegressor(n_estimators=num)
        regr.fit(x_train, y_train)
        test_scores.append(regr.score(x_test, y_test))
        train_scores.append(regr.score(x_train, y_train))
    ax.plot(nums, test_scores, label="test_score")
    ax.plot(nums, train_scores, label="train_score")
    ax.set_xlabel("num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    plt.show()


# 考虑到个体回归树的深度的影响
def test_GradientBoostingRegressor_maxdepth(*data):
    x_train, x_test, y_train, y_test = data
    maxdepths = np.arange(1, 20)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for maxdepth in maxdepths:
        regr = ensemble.GradientBoostingRegressor(max_depth=maxdepth, max_leaf_nodes=None)
        regr.fit(x_train, y_train)
        test_scores.append(regr.score(x_test, y_test))
        train_scores.append(regr.score(x_train, y_train))
    ax.plot(maxdepths, test_scores, label="test_score")
    ax.plot(maxdepths, train_scores, label="train_score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    plt.show()


# 考虑到学习率的影响
def test_GradientBoostingRegressor_learning(*data):
    x_train, x_test, y_train, y_test = data
    learning_rates = np.linspace(0.01, 1.0)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for learning_rate in learning_rates:
        regr = ensemble.GradientBoostingRegressor(learning_rate=learning_rate)
        regr.fit(x_train, y_train)
        test_scores.append(regr.score(x_test, y_test))
        train_scores.append(regr.score(x_train, y_train))
    ax.plot(learning_rates, test_scores, label="test_score")
    ax.plot(learning_rates, train_scores, label="train_score")
    ax.set_xlabel("learning_rates")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    plt.show()


# 考虑到subsample的影响
def test_GradientBoostingRegressor_subsample(*data):
    x_train, x_test, y_train, y_test = data
    subsamples = np.linspace(0.01, 1.0, num=20)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for subsample in subsamples:
        regr = ensemble.GradientBoostingRegressor(subsample=subsample)
        regr.fit(x_train, y_train)
        test_scores.append(regr.score(x_test, y_test))
        train_scores.append(regr.score(x_train, y_train))
    ax.plot(subsamples, test_scores, label="test_score")
    ax.plot(subsamples, train_scores, label="train_score")
    ax.set_xlabel("subsamples")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    plt.show()


# 考虑到max_features的影响
def test_GradientBoostingRegressor_max_features(*data):
    x_train, x_test, y_train, y_test = data
    max_features = np.linspace(0.01, 1.0)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for max_feature in max_features:
        regr = ensemble.GradientBoostingRegressor(max_features=max_feature)
        regr.fit(x_train, y_train)
        test_scores.append(regr.score(x_test, y_test))
        train_scores.append(regr.score(x_train, y_train))
    ax.plot(max_features, test_scores, label="test_score")
    ax.plot(max_features, train_scores, label="train_score")
    ax.set_xlabel("max_features")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    plt.show()


# loss函数的影响
def test_GradientBoostingRegressor_loss(*data):
    x_train, x_test, y_train, y_test = data
    fig = plt.figure()
    nums = np.arange(1, 200, 2)
    losses = ['ls', 'lad', 'huber']
    ax = fig.add_subplot(2, 1, 1)
    alphas = np.linspace(0.01, 1.0, endpoint=False, num=5)
    for alpha in alphas:
        testing_scores = []
        training_scores = []
        for num in nums:
            regr = ensemble.GradientBoostingRegressor(n_estimators=num, loss='huber', alpha=alpha)
            regr.fit(x_train, y_train)
            training_scores.append(regr.score(x_train, y_train))
            testing_scores.append(regr.score(x_test, y_test))
        ax.plot(nums, training_scores, label="train_score:%f" % alpha)
        ax.plot(nums, testing_scores, label="test_score:%f" % alpha)
    ax.set_xlabel("num")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    ax.set_title("loss:huber")
    ax = fig.add_subplot(2, 1, 2)
    for loss in ['ls', 'lad']:
        testing_scores = []
        training_scores = []
        for num in nums:
            regr = ensemble.GradientBoostingRegressor(n_estimators=num, loss=loss)
            regr.fit(x_train, y_train)
            training_scores.append(regr.score(x_train, y_train))
            testing_scores.append(regr.score(x_test, y_test))
        ax.plot(nums, training_scores, label="train_score:%s" % loss)
        ax.plot(nums, testing_scores, label="test_score:%s" % loss)
    ax.set_xlabel("num")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    ax.set_title("loss:ls/lad")
    plt.show()


# 随机森林/RandomForest
def test_RandomForest(*data):
    x_train, x_test, y_train, y_test = data
    clf = ensemble.RandomForestClassifier()
    clf.fit(x_train, y_train)
    print("train_score:", clf.score(x_train, y_train))
    print("test_score:", clf.score(x_test, y_test))


# 森林中的决策树的个数的影响
def test_RandomForest_num(*data):
    x_train, x_test, y_train, y_test = data
    nums = np.arange(1, 100, 2)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for num in nums:
        clf = ensemble.RandomForestClassifier(n_estimators=num)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_test, y_test))
        train_scores.append(clf.score(x_train, y_train))
    ax.plot(nums, test_scores, label="test_score")
    ax.plot(nums, train_scores, label="train_score")
    ax.set_xlabel("num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    plt.show()


# max_depth参数的影响
def test_RandomForest_max_depth(*data):
    x_train, x_test, y_train, y_test = data
    maxdepths = range(1, 20)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for maxdepth in maxdepths:
        clf = ensemble.RandomForestClassifier(max_depth=maxdepth)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_test, y_test))
        train_scores.append(clf.score(x_train, y_train))
    ax.plot(maxdepths, test_scores, label="test_score")
    ax.plot(maxdepths, train_scores, label="train_score")
    ax.set_xlabel("num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    plt.show()


# 考虑到max_features的影响
def test_RandomForest_max_features(*data):
    x_train, x_test, y_train, y_test = data
    max_features = np.linspace(0.01, 1.0)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for max_feature in max_features:
        clf = ensemble.RandomForestClassifier(max_features=max_feature)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_test, y_test))
        train_scores.append(clf.score(x_train, y_train))
    ax.plot(max_features, test_scores, label="test_score")
    ax.plot(max_features, train_scores, label="train_score")
    ax.set_xlabel("num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    plt.show()


# 随机森林回归器
def test_RandomForestRegressor(*data):
    x_train, x_test, y_train, y_test = data
    regr = ensemble.RandomForestRegressor()
    regr.fit(x_train, y_train)
    print("train_score:", regr.score(x_train, y_train))
    print("test_score:", regr.score(x_test, y_test))


# 回归树棵树
def test_RandomForestRegressor_num(*data):
    x_train, x_test, y_train, y_test = data
    nums = np.arange(1, 100, 2)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for num in nums:
        regr = ensemble.RandomForestRegressor(n_estimators=num)
        regr.fit(x_train, y_train)
        test_scores.append(regr.score(x_test, y_test))
        train_scores.append(regr.score(x_train, y_train))
    ax.plot(nums, test_scores, label="test_score")
    ax.plot(nums, train_scores, label="train_score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    plt.show()


# max_depth
def test_RandomForestRegressor_max_depths(*data):
    x_train, x_test, y_train, y_test = data
    max_depths = range(1, 20)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for max_depth in max_depths:
        regr = ensemble.RandomForestRegressor(max_depth=max_depth)
        regr.fit(x_train, y_train)
        test_scores.append(regr.score(x_test, y_test))
        train_scores.append(regr.score(x_train, y_train))
    ax.plot(max_depths, test_scores, label="test_score")
    ax.plot(max_depths, train_scores, label="train_score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    plt.show()


# max_features
def test_RandomForestRegressor_max_features(*data):
    x_train, x_test, y_train, y_test = data
    max_features = np.linspace(0.01, 1.0)
    fig = plt.figure()
    train_scores = []
    test_scores = []
    ax = fig.add_subplot(1, 1, 1)
    for max_feature in max_features:
        regr = ensemble.RandomForestRegressor(max_features=max_feature)
        regr.fit(x_train, y_train)
        test_scores.append(regr.score(x_test, y_test))
        train_scores.append(regr.score(x_train, y_train))
    ax.plot(max_features, test_scores, label="test_score")
    ax.plot(max_features, train_scores, label="train_score")
    ax.set_xlabel("max_features")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc='best')
    plt.show()