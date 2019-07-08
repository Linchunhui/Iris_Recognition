import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from cnn_feature import *
from tabulate import tabulate  # 13min+
import matplotlib.pyplot as plt

train = pd.read_csv("D:/study/iris/csv/train.csv")
test = pd.read_csv("D:/study/iris/csv/test.csv")
train_vector = pd.read_csv("D:/study/iris/csv/train_vector1.csv")
test_vector = pd.read_csv("D:/study/iris/csv/test_vector1.csv")

train_img_list = train["img"]
train_label_list = train["label"]
test_img_list = test["img"]
test_label_list = test["label"]
train_size = len(train_img_list)
test_size = len(test_img_list)

train_features = np.array(train_vector)
train_classes = train_label_list
test_features = np.array(test_vector)
test_classes = test_label_list

def onehot(train_classes):
    train_classes = np.array(train_classes).reshape(-1, 1)
    enc = OneHotEncoder(categories='auto')
    enc.fit(train_classes)

    # one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
    tempdata = enc.transform(train_classes).toarray()
    return tempdata

def SVM(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))

    pca = PCA(n_components=680, whiten=True, random_state=42)
    #svc = SVC(kernel='rbf', C=8, gamma=0.001)
    svc = KNeighborsClassifier(n_neighbors=1,metric="cosine")
    svm = make_pipeline(pca, svc)
    svm.fit(train_redfeatures, train_classes)
    svmclasses = svm.predict(test_redfeatures)
    svmcrr = float(np.sum(svmclasses == test_classes)) / total
    return (svmcrr*100)

def SVM1(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))

    #lle = LocallyLinearEmbedding(n_neighbors=200 + 1, n_components=200)
    #lle.fit(train_features)
    #train_redfeatures = lle.transform(train_redfeatures)
    #test_redfeatures = lle.transform(test_redfeatures)
    #print("finish reduce dim!")
    n = range(220, 601, 20)
    l1_crr = []
    for i in range(len(n)):
        pca = PCA(n_components=n[i], whiten=True, random_state=42)
        svc = SVC(kernel='rbf', C=5, gamma=0.001)

        svm = make_pipeline(pca, svc)
        svm.fit(train_redfeatures, train_classes)
        svmclasses = svm.predict(test_redfeatures)
        svmcrr = float(np.sum(svmclasses == test_classes)) / total
        print(n[i],svmcrr)
        l1_crr.append(svmcrr)
    print(
        tabulate([[n[0], l1_crr[0]],
                  [n[1], l1_crr[1]],
                  [n[2], l1_crr[2]],
                  [n[3], l1_crr[3]],
                  [n[4], l1_crr[4]],
                  [n[5], l1_crr[5]],
                  [n[6], l1_crr[6]],
                  [n[7], l1_crr[7]],
                  [n[8], l1_crr[8]],
                  [n[9], l1_crr[9]],
                  [n[10], l1_crr[10]],
                  [n[11], l1_crr[11]],
                  [n[12], l1_crr[12]],
                  [n[13], l1_crr[13]],
                  [n[14], l1_crr[14]]],
                 # [n[15], l1_crr[15]]],
                 headers=['Dimensionality of the feature vector', 'SVM distance measure']))
    plt.plot(n, l1_crr, marker="*", color='navy')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition Rate')
    plt.savefig('D:/study/iris/fig/svm_dense_layer2_12.png')
    plt.show()

    #return l1_crr

def KNN(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))

    pca = PCA(n_components=150, whiten=True, random_state=42)
    cosknn1 = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    cosknn=make_pipeline(pca, cosknn1)

    cosknn.fit(train_redfeatures, train_classes)
    cosclasses = cosknn.predict(test_redfeatures)
    coscrr = float(np.sum(cosclasses == test_classes)) / total
    return (coscrr*100)


def DTC(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))
    train_classes=onehot(train_classes)
    pca = PCA(n_components=160, whiten=True, random_state=42)
    dtc = DecisionTreeClassifier(max_depth=19,min_samples_leaf=12,min_samples_split=10)
    model = make_pipeline(pca, dtc)

    model.fit(train_redfeatures, train_classes)
    classes = model.predict(test_redfeatures)
    crr = float(np.sum(classes == test_classes)) / total
    #crr = model.score(test_redfeatures,test_classes)
    return crr

def RF(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))

    #pca = PCA(n_components=160, whiten=True, random_state=42)
    model = RandomForestClassifier(n_estimators=800, max_features=250, max_depth=15, min_samples_split=120,
                                  min_samples_leaf=25 ,oob_score=True, random_state=10)
    #model = make_pipeline(pca, rf)

    model.fit(train_redfeatures, train_classes)
    classes = model.predict(test_redfeatures)
    crr = float(np.sum(classes == test_classes)) / total
    crr = model.score(test_redfeatures,test_classes)
    return crr

def Ada(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))

    pca = PCA(n_components=160, whiten=True, random_state=42)
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                         algorithm="SAMME.R",#可以不写
                         n_estimators=200)
    model = make_pipeline(pca, ada)

    model.fit(train_redfeatures, train_classes)
    classes = model.predict(test_redfeatures)
    crr = float(np.sum(classes == test_classes)) / total
    #crr = model.score(test_redfeatures,test_classes)
    return crr
if __name__=="__main__":
    #PE.table_CRR(train_features, train_classes, test_features, test_classes)
    #PE.performance_evaluation1(train_features, train_classes, test_features, test_classes)
    init_tf()
    #init_tf_dense()
    #inceptionv4_init()
    train_features1 = ex_train()
    test_features1 = ex_test()
    print("train fe",train_features1.shape)
    print("train label",train_classes.shape)
    print("test fe",test_features1.shape)
    print("test label",test_classes.shape)
    print(SVM(train_features1, train_classes, test_features1, test_classes))
    #SVM1(train_features1, train_classes, test_features1, test_classes)
   # print(DTC(train_features, train_classes, test_features, test_classes))
