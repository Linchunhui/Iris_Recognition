from Matching import IrisMatching, IrisMatchingRed, IrisMatchingRed1,calcROC
from tabulate import tabulate  # 13min+
import matplotlib.pyplot as plt
import numpy as np

thresholds_2 = [0.076, 0.085, 0.1]


def table_CRR(train_features, train_classes, test_features, test_classes):
    thresholds = np.arange(0.04, 0.1, 0.003)
    L1_1, _, _ = IrisMatching(train_features, train_classes, test_features, test_classes, 1)
    print("L1:",(L1_1*100))
    L2_1, _, _ = IrisMatching(train_features, train_classes, test_features, test_classes, 2)
    print("L2:",(L2_1*100))
    C_1, distsm, distsn = IrisMatching(train_features, train_classes, test_features, test_classes, 3)
    print("Cos:",(C_1*100))
    #L1_2, L2_2, C_2 = IrisMatchingRed(train_features, train_classes, test_features, test_classes, 280)
    print("Correct recognition rate (%)")
    print(
    tabulate([['L1 distance measure', L1_1 * 100], ['L2 distance measure', L2_1 * 100],
              ['Cosine similarity measure', C_1 * 100]],
             headers=['Similartiy measure', 'Original feature set']))
    fmrs, fnmrs = calcROC(distsm, distsn, thresholds)
    plt.plot(fmrs, fnmrs)
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non_match Rate')
    plt.title('ROC Curve')
    plt.savefig('D:/study/iris/fig/roc_curve1.png')
    plt.show()


# table_CRR(train_features, train_classes, test_features, test_classes)

def performance_evaluation(train_features, train_classes, test_features, test_classes):
    n = range(140, 441, 20)
    l1_crr=[]
    l2_crr=[]
    cos_crr = []
    for i in range(len(n)):
        l1crr, l2crr, coscrr = IrisMatchingRed(train_features, train_classes, test_features, test_classes, n[i])
        print(n[i],l1crr*100,l2crr*100,coscrr*100)
        l1_crr.append(l1crr * 100)
        l2_crr.append(l2crr * 100)
        cos_crr.append(coscrr * 100)
    print(
        tabulate([[n[0],l1_crr[0], l2_crr[0], cos_crr[0]],
                 [n[1], l1_crr[1], l2_crr[1], cos_crr[1]],
                 [n[2], l1_crr[2], l2_crr[2], cos_crr[2]],
                 [n[3], l1_crr[3], l2_crr[3], cos_crr[3]],
                 [n[4], l1_crr[4], l2_crr[4], cos_crr[4]],
                 [n[5], l1_crr[5], l2_crr[5], cos_crr[5]],
                 [n[6], l1_crr[6], l2_crr[6], cos_crr[6]],
                 [n[7], l1_crr[7], l2_crr[7], cos_crr[7]],
                 [n[8], l1_crr[8], l2_crr[8], cos_crr[8]],
                 [n[9], l1_crr[9], l2_crr[9], cos_crr[9]],
                 [n[10], l1_crr[10], l2_crr[10], cos_crr[10]],
                 [n[11], l1_crr[11], l2_crr[11], cos_crr[11]],
                 [n[12], l1_crr[12], l2_crr[12], cos_crr[12]],
                 [n[13], l1_crr[13], l2_crr[13], cos_crr[13]],
                 [n[14], l1_crr[14], l2_crr[14], cos_crr[14]],
                 [n[15], l1_crr[15], l2_crr[15], cos_crr[15]]],
                 headers=['Dimensionality of the feature vector', 'L1 distance measure', "L2 distance measure","Cosine similarity measure"]))
    plt.plot(n, l1_crr, marker="*", color='navy')
    plt.plot(n, l2_crr, marker="*", color='blue')
    plt.plot(n, cos_crr, marker="*", color='red')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition Rate')
    plt.savefig('D:/study/iris/fig/figure_reduce11.png')
    plt.show()

def performance_evaluation1(train_features, train_classes, test_features, test_classes):
    n = range(140, 441, 20)
    l1_crr=[]
    l2_crr=[]
    cos_crr = []
    for i in range(len(n)):
        l1crr = IrisMatchingRed1(train_features, train_classes, test_features, test_classes, n[i])
        print(n[i],l1crr*100)
        l1_crr.append(l1crr * 100)
    print(
        tabulate([[n[0],l1_crr[0]],
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
                 [n[14], l1_crr[14]],
                 ["none", l1_crr[15]]],
                 headers=['Dimensionality of the feature vector', 'SVM distance measure']))
    plt.plot(n, l1_crr, marker="*", color='navy')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition Rate')
    plt.savefig('D:/study/iris/fig/svm_reduce12.png')
    plt.show()

# performance_evaluation(train_features, train_classes, test_features, test_classes)


def FM_FNM_table(fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u, thresholds):
    print("False Match and False Nonmatch Rates with Different Threshold Values")
    print(
    tabulate([[thresholds[7], str(fmrs_mean[7]) + "[" + str(fmrs_l[7]) + "," + str(fmrs_u[7]) + "]",
               str(fnmrs_mean[7]) + "[" + str(fnmrs_l[7]) + "," + str(fnmrs_u[7]) + "]"],
              [thresholds[8], str(fmrs_mean[8]) + "[" + str(fmrs_l[8]) + "," + str(fmrs_u[8]) + "]",
               str(fnmrs_mean[8]) + "[" + str(fnmrs_l[8]) + "," + str(fnmrs_u[8]) + "]"],
              [thresholds[9], str(fmrs_mean[9]) + "[" + str(fmrs_l[9]) + "," + str(fmrs_u[9]) + "]",
               str(fnmrs_mean[9]) + "[" + str(fnmrs_l[9]) + "," + str(fnmrs_u[9]) + "]"]],
             headers=['Threshold', 'False match rate(%)', "False non-match rate(%)"]))


# FM_FNM_table(train_features, train_classes, test_features, test_classes, thresholds_2)

def FMR_conf(fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u):
    plt.figure()
    lw = 2
    plt.plot(fmrs_mean, fnmrs_mean, color='navy', lw=lw, linestyle='-')
    plt.plot(fmrs_l, fnmrs_mean, color='navy', lw=lw, linestyle='--')
    plt.plot(fmrs_u, fnmrs_mean, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 60])
    plt.ylim([0.0, 40])
    plt.xlabel('False Match Rate(%)')
    plt.ylabel('False Non_match Rate(%)')
    plt.title('FMR Confidence Interval')
    plt.savefig('D:/study/iris/fig/figure_13_a.png')
    plt.show()


def FNMR_conf(fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u):
    plt.figure()
    lw = 2
    plt.plot(fmrs_mean, fnmrs_mean, color='navy', lw=lw, linestyle='-')
    plt.plot(fmrs_mean, fnmrs_l, color='navy', lw=lw, linestyle='--')
    plt.plot(fmrs_mean, fnmrs_u, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 100])
    plt.ylim([0.0, 40])
    plt.xlabel('False Match Rate(%)')
    plt.ylabel('False Non_match Rate(%)')
    plt.title('FNMR Confidence Interval')
    plt.savefig('D:/study/iris/fig/figure_13_b.png')
    plt.show()

# FMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u)
