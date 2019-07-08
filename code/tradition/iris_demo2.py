#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from Gabor import FeatureExtraction
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
import Evaluation as PE
from tabulate import tabulate  # 13min+
import matplotlib.pyplot as plt
train = pd.read_csv("D:/study/iris/csv/train.csv")
test = pd.read_csv("D:/study/iris/csv/test.csv")

train_img_list = train["img"]
train_label_list = train["label"]
test_img_list = test["img"]
test_label_list = test["label"]
train_size = len(train_img_list)
test_size = len(test_img_list)

#train_features = np.zeros((train_size,-1))
train_classes = np.zeros(train_size, dtype = np.uint8)
#test_features = np.zeros((test_size,-1))
test_classes = np.zeros(test_size, dtype = np.uint8)

#starttime = datetime.datetime.now()
dir1 = r"D:\study\iris\CASIA\enhance\train"
dir2 = r"D:\study\iris\CASIA\enhance\test"
train=[]
test=[]
for i in range(train_size):
    train_path = os.path.join(dir1, train_img_list[i].split("\\")[-1])
    img = cv2.imread(train_path, 0)
    train.append(FeatureExtraction(img))
    print("train:", i)
    train_classes[i] = train_label_list[i]
train_features = np.array(train)
print(train_features.shape)

for j in range(test_size):
    test_path = os.path.join(dir2, test_img_list[j].split("\\")[-1])
    img = cv2.imread(test_path, 0)
    test.append(FeatureExtraction(img))
    print("test:",j)
    test_classes[j] = test_label_list[j]
test_features = np.array(test)
print(test_features.shape)
#endtime = datetime.datetime.now()
def SVM(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))

    pca = PCA(n_components=380, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', C=5, gamma=0.001)

    svm = make_pipeline(pca, svc)
    svm.fit(train_redfeatures, train_classes)
    svmclasses = svm.predict(test_redfeatures)
    svmcrr = float(np.sum(svmclasses == test_classes)) / total
    return svmcrr
def KNN(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))

    pca = PCA(n_components=380, whiten=True, random_state=42)
    cosknn1 = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    cosknn=make_pipeline(pca, cosknn1)

    cosknn.fit(train_redfeatures, train_classes)
    cosclasses = cosknn.predict(test_redfeatures)
    coscrr = float(np.sum(cosclasses == test_classes)) / total
    return coscrr

def SVM1(train_features, train_classes, test_features, test_classes):
    train_redfeatures = train_features.copy()
    test_redfeatures = test_features.copy()
    total = float(len(test_classes))

    #lle = LocallyLinearEmbedding(n_neighbors=200 + 1, n_components=200)
    #lle.fit(train_features)
    #train_redfeatures = lle.transform(train_redfeatures)
    #test_redfeatures = lle.transform(test_redfeatures)
    #print("finish reduce dim!")
    n = range(220, 501, 20)
    l1_crr = []
    for i in range(len(n)):
        pca = PCA(n_components=n[i], whiten=True, random_state=42)
        svc = SVC(kernel='rbf', C=5, gamma=0.001)

        svm = make_pipeline(pca, svc)
        svm.fit(train_redfeatures, train_classes)
        svmclasses = svm.predict(test_redfeatures)
        svmcrr = float(np.sum(svmclasses == test_classes)) / total
        print("svm",n[i],svmcrr)
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
    plt.savefig('D:/study/iris/fig/svm_gabor.png')
    plt.show()

def KNN1(train_features, train_classes, test_features, test_classes):
        train_redfeatures = train_features.copy()
        test_redfeatures = test_features.copy()
        total = float(len(test_classes))

        n = range(220, 501, 20)
        l1_crr = []
        for i in range(len(n)):
            pca = PCA(n_components=n[i], whiten=True, random_state=42)
            cosknn1 = KNeighborsClassifier(n_neighbors=1, metric='cosine')

            cosknn = make_pipeline(pca,cosknn1)
            cosknn.fit(train_redfeatures, train_classes)
            knnclasses = cosknn.predict(test_redfeatures)
            knncrr = float(np.sum(knnclasses == test_classes)) / total
            print("knn:",n[i], knncrr)
            l1_crr.append(knncrr)
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
                     headers=['Dimensionality of the feature vector', 'KNN distance measure']))
        plt.plot(n, l1_crr, marker="*", color='navy')
        plt.xlabel('Dimensionality of the feature vector')
        plt.ylabel('Correct Recognition Rate')
        plt.savefig('D:/study/iris/fig/knn_gabor.png')
        plt.show()

    #return l1_crr
#print('image processing and feature extraction takes '+str((endtime-starttime).seconds)+' seconds'
if __name__=="__main__":
    #PE.table_CRR(train_features, train_classes, test_features, test_classes)
    #print("KNN:",KNN(train_features, train_classes, test_features, test_classes))
    #print("SVM:",SVM(train_features, train_classes, test_features, test_classes))
    #SVM1(train_features, train_classes, test_features, test_classes)
    #KNN1(train_features, train_classes, test_features, test_classes)
    PE.table_CRR(train_features, train_classes, test_features, test_classes)
#thresholds_2=[0.74,0.76,0.78]