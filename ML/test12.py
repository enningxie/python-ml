# 模型评估/选择与验证


# 损失函数
def test_loss1():
    from sklearn.metrics import zero_one_loss
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    print("zero_one_loss<fraction>", zero_one_loss(y_true, y_pred, normalize=True))
    print("zero_one_loss<num>", zero_one_loss(y_true, y_pred, normalize=False))


# 对数损失函数
def test_loss2():
    from sklearn.metrics import log_loss
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.9, 0.1]
    ]
    print("log_loss<average>:", log_loss(y_true, y_pred, normalize=True))
    print("log_loss<total>:", log_loss(y_true, y_pred, normalize=False))


# 数据集切分
# 分层采样保证了训练集和测试集中的各类样本的比例与原始数据集一致
def test_split1():
    from sklearn.model_selection import train_test_split
    x = [
        [1, 2, 3, 4],
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74]
    ]
    y = [1, 1, 0, 0, 1, 1, 0, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    print("x_train:", x_train)
    print("x_test:", x_test)
    print("y_train:", y_train)
    print("y_test:", y_test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0, stratify=y)
    print("x_train_:", x_train)
    print("x_test_:", x_test)
    print("y_train_:", y_train)
    print("y_test_:", y_test)


# kfold k折交叉切分
def test_split2():
    from sklearn.model_selection import KFold
    import numpy as np
    x = np.array([
        [1, 2, 3, 4],
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74],
        [81, 82, 83, 84]
    ])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1])
    folder = KFold(random_state=0, shuffle=False, n_splits=3)
    for train_index, test_index in folder.split(x, y):
        print("train_index:", train_index)
        print("test_index:", test_index)
        print("train_:", x[train_index])
        print("train_:", x[test_index])
    folder_ = KFold(random_state=0, shuffle=True, n_splits=3)
    for train_index, test_index in folder_.split(x, y):
        print("train_index_:", train_index)
        print("test_index_:", test_index)
        print("train__:", x[train_index])
        print("train__:", x[test_index])


# stratifieldKFold 实现了数据集的分层采样k折交叉切分
def test_split3():
    from sklearn.model_selection import KFold, StratifiedKFold
    import numpy as np
    x = np.array([
        [1, 2, 3, 4],
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74]
    ])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    folder = KFold(random_state=0, shuffle=False, n_splits=4)
    for train_index, test_index in folder.split(x, y):
        print("train_index:", train_index)
        print("test_index:", test_index)
        print("train_:", x[train_index])
        print("train_:", x[test_index])
    folder_ = StratifiedKFold(random_state=0, shuffle=False, n_splits=4)
    for train_index, test_index in folder_.split(x, y):
        print("train_index_:", train_index)
        print("test_index_:", test_index)
        print("train__:", x[train_index])
        print("train__:", x[test_index])


# 留一法LeaveOneOut
def test_split4():
    from sklearn.model_selection import LeaveOneOut
    import numpy as np
    x = np.array([
        [1, 2, 3, 4],
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34]
    ])
    y = np.array([1, 1, 0, 0])
    lo = LeaveOneOut()
    for train_index, test_index in lo.split(x, y):
        print("train_index_:", train_index)
        print("test_index_:", test_index)
        print("train__:", x[train_index])
        print("train__:", x[test_index])


# cross_val_score 在指定数据集上运行指定学习器时，通过k折交叉获取的最佳性能
def test_split5():
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_digits
    from sklearn.svm import LinearSVC

    digits = load_digits()
    x = digits.data
    y = digits.target

    result = cross_val_score(LinearSVC(), x, y, cv=10)
    print("cross_val_score:", result)


# 性能度量
# 分类问题的性能度量
# accuracy_score/用于计算分类结果的准确率（分类正确的比例）
def test_score1():
    from sklearn.metrics import accuracy_score
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    print("Accuracy score(normalize=True):", accuracy_score(y_true, y_pred, normalize=True))
    print("Accuracy score(normalize=False):", accuracy_score(y_true, y_pred, normalize=False))


# precision_score/用于计算分类结果的查准率（预测结果为正类的那些样本中，有多少比例确实是正类）
def test_score2():
    from sklearn.metrics import accuracy_score, precision_score
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    print("Accuracy score:", accuracy_score(y_true, y_pred, normalize=True))
    print("precision score:", precision_score(y_true, y_pred))


# recall_score/用于计算结果的查全率（真实的正类中，有多少比例被预测为正类）
def test_score3():
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    print("Accuracy score:", accuracy_score(y_true, y_pred, normalize=True))
    print("precision score:", precision_score(y_true, y_pred))
    print("recall score:", recall_score(y_true, y_pred))


# f1_score（查准率和查全率的调和均值）
def test_score4():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    print("Accuracy score:", accuracy_score(y_true, y_pred, normalize=True))
    print("precision score:", precision_score(y_true, y_pred))
    print("recall score:", recall_score(y_true, y_pred))
    print("f1 score:", f1_score(y_true, y_pred))


# fbeta_score（）
def test_score5():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    print("Accuracy score:", accuracy_score(y_true, y_pred, normalize=True))
    print("precision score:", precision_score(y_true, y_pred))
    print("recall score:", recall_score(y_true, y_pred))
    print("f1 score:", f1_score(y_true, y_pred))
    print("fbeta score(0.001):", fbeta_score(y_true, y_pred, beta=0.001))
    print("fbeta score(1):", fbeta_score(y_true, y_pred, beta=1))
    print("fbeta score(10):", fbeta_score(y_true, y_pred, beta=10))
    print("fbeta score(10000):", fbeta_score(y_true, y_pred, beta=10000))


# classification_report 返回性能指标的字符串
def test_score6():
    from sklearn.metrics import classification_report
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    print("classification Report:", classification_report(y_true, y_pred, target_names=["class0", "class1"]))


# confusion_matrix (分类结果的混淆矩阵)
def test_score7():
    from sklearn.metrics import confusion_matrix
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))


# precision_recall_curve 用于计算分类结果的P-R曲线
def test_score8():
    from sklearn.metrics import precision_recall_curve
    from sklearn.datasets import load_iris
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize
    import numpy as np

    # 加载数据
    iris = load_iris()
    x = iris.data
    y = iris.target
    # 二元化标记
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    # 添加噪声
    np.random.seed(0)
    n_samples, n_features = x.shape
    x = np.c_[x, np.random.randn(n_samples, 200*n_features)]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    # 训练模型
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0))
    clf.fit(x_train, y_train)
    y_score = clf.decision_function(x_test)
    # 获取P-R
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        ax.plot(recall[i], precision[i], label="target=%s" % i)
    ax.set_xlabel("Recall Score")
    ax.set_ylabel("Precision Score")
    ax.legend(loc='best')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid()
    plt.show()


# roc_curve 用于计算分类结果的ROC曲线/有点问题todo
def test_score9():
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.datasets import load_iris
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize
    import numpy as np

    # 加载数据
    iris = load_iris()
    x = iris.data
    y = iris.target
    # 二元化标记
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    # 添加噪声
    np.random.seed(0)
    n_samples, n_features = x.shape
    x = np.c_[x, np.random.randn(n_samples, 200 * n_features)]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    # 训练模型
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0))
    clf.fit(x_train, y_train)
    y_score = clf.decision_function(x_test)

    # 获取P-R
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], label="target=%s,auc=%s" % (i, roc_auc[i]))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc='best')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid()
    plt.show()


# 回归问题的性能度量
# 计算回归预测误差的绝对值的平均值
def test_regr_score1():
    from sklearn.metrics import mean_absolute_error
    y_true = [1, 1, 1, 1, 1, 2, 2, 2, 0, 0]
    y_pred = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    print("Mean Absolute Error:", mean_absolute_error(y_true, y_pred))


# 计算回归预测误差的平方的平均值
def test_regr_score2():
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_true = [1, 1, 1, 1, 1, 2, 2, 2, 0, 0]
    y_pred = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    print("Mean Absolute Error:", mean_absolute_error(y_true, y_pred))
    print("Mean squared Error:", mean_squared_error(y_true, y_pred))


# 验证曲线
def test_regr_score3():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import validation_curve  # 验证曲线

    # 加载数据
    digits = load_digits()
    x, y = digits.data, digits.target
    # 获取验证曲线
    param_name = 'C'
    param_range = np.logspace(-2, 2)
    train_scores, test_scores = validation_curve(LinearSVC(), x, y, param_name=param_name, param_range=param_range, cv=10, scoring="accuracy")
    # 对每个c，获取10折交叉上的预测得分上的均值和方差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(param_range, train_scores_mean, label="Training Accuracy", color='r')
    ax.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='r')
    ax.semilogx(param_range, test_scores_mean, label="Testing Accuracy", color='g')
    ax.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                    color='g')
    ax.set_xlabel("c")
    ax.set_ylabel("score")
    ax.legend(loc='best')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid()
    plt.show()


# 学习曲线
def test_regr_score4():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import learning_curve  # 学习曲线

    # 加载数据
    digits = load_digits()
    x, y = digits.data, digits.target
    # 获取验证曲线
    train_sizes = np.linspace(0.1, 1.0, endpoint=True, dtype='float')
    abs_train_scores, train_scores, test_scores = learning_curve(LinearSVC(), x, y, cv=10, scoring="accuracy", train_sizes=train_sizes)
    # 对每个c，获取10折交叉上的预测得分上的均值和方差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(abs_train_scores, train_scores_mean, label="Training Accuracy", color='r')
    ax.fill_between(abs_train_scores, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                    color='r')
    ax.semilogx(abs_train_scores, test_scores_mean, label="Testing Accuracy", color='g')
    ax.fill_between(abs_train_scores, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                    color='g')
    ax.set_xlabel("sample")
    ax.set_ylabel("score")
    ax.legend(loc='best')
    ax.set_ylim(0, 1.1)
    ax.grid()
    plt.show()


# 参数优化
# 自动调参进行参数优化是使用sklearn优雅地进行机器学习的核心，自动化调参技术帮我们省去了人工调参的繁琐和经验不足
# 暴力搜索寻优 GridSearchCV
# 网格搜索
def test_search1():
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    # 加载数据
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0, stratify=digits.target)
    # 参数优化
    tuned_parameters = [{'penalty': ['l1', 'l2'],
                         'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                         'solver': ['liblinear'],
                         'multi_class': ['ovr']},
                        {'penalty': ['l2'],
                         'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                         'solver': ['lbfgs'],
                         'multi_class': ['ovr', 'multinomial']}
                        ]
    clf = GridSearchCV(LogisticRegression(tol=1e-6), tuned_parameters, cv=10)
    clf.fit(x_train, y_train)
    print("Best parameters set found:", clf.best_params_)
    print("Grid scores:")
    for params, mean_score, scores in clf.grid_scores_:
        print("\t%0.3f (+/-%0.03f) for %s" % (mean_score, scores.std() * 2, params))
    print("Optimized Score:", clf.score(x_test, y_test))
    print("Detailed classification report:")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))


# 随机搜索寻优
def test_search2():
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    import scipy
    # 加载数据
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0, stratify=digits.target)
    # 参数优化
    tuned_parameters = {
        'C': scipy.stats.expon(scale=100),
        'multi_class': ['ovr', 'multinomial']
    }
    clf = RandomizedSearchCV(LogisticRegression(penalty='l2', solver='lbfgs', tol=1e-6), tuned_parameters, cv=10, scoring='accuracy', n_iter=100)
    clf.fit(x_train, y_train)
    print("Best parameters set found:", clf.best_params_)
    print("Random Grid scores:")
    for params, mean_score, scores in clf.grid_scores_:
        print("\t%0.3f (+/-%0.03f) for %s" % (mean_score, scores.std() * 2, params))
    print("Optimized Score:", clf.score(x_test, y_test))
    print("Detailed classification report:")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))


