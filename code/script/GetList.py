import pandas as pd
import numpy as np
import os, glob

def get_label():
    list1 = [i for i in range(1,109)]
    label = []
    for i in range(len(list1)):
        if len(str(list1[i])) == 1:
            s = "00" + str(list1[i])
            label.append(s)
            continue
        if len(str(list1[i])) == 2:
            s = "0" + str(list1[i])
            label.append(s)
            continue
        else:
            label.append(str(list1[i]))
    return label

def get_LR():
    return ['L','R']

def get_onetwo():
    return ['1','2']

def get_files(file_dir):
    label = get_label()
    LR=get_LR()
    image_list, label_list = [], []
    for i in label:
        for j in LR:
            for img in glob.glob(os.path.join(file_dir, i, j, "*.jpg")):
                image_list.append(img)
                label_list.append(int(i))
    print('There are %d data' %(len(image_list)))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

def split(img_list,label_list,split_rate=0.3):
    l = len(img_list)*(1-split_rate)
    l = int(l)
    #print(l)
    img_train_list = img_list[:l]
    label_train_list = label_list[:l]
    img_test_list = img_list[l:]
    label_test_list = label_list[l:]
    return img_train_list, label_train_list, img_test_list, label_test_list

def get_files1(file_dir):
    label = get_label()
    LR=get_onetwo()
    image_list_train, label_list_train, image_list_test, label_list_test = [], [], [], []
    for i in label:
        for img in glob.glob(os.path.join(file_dir, i, LR[0], "*.bmp")):
            image_list_train.append(img)
            label_list_train.append(int(i))
        for img in glob.glob(os.path.join(file_dir, i, LR[1], "*.bmp")):
            image_list_test.append(img)
            label_list_test.append(int(i))
    print('There are %d data' %(len(image_list_train)))
    print('There are %d data' % (len(image_list_test)))

    temp = np.array([image_list_train, label_list_train])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list_train = list(temp[:, 0])
    label_list_train = list(temp[:, 1])
    label_list_train = [int(i) for i in label_list_train]

    temp1 = np.array([image_list_test, label_list_test])
    temp1 = temp1.transpose()
    np.random.shuffle(temp1)
    image_list_test = list(temp1[:, 0])
    label_list_test = list(temp1[:, 1])
    label_list_test = [int(i) for i in label_list_test]
    return image_list_train, label_list_train, image_list_test, label_list_test
def main():
    file_dir = "D:/study/iris/CASIA-Iris-Thousand"
    img_list, label_list = get_files(file_dir)
    img_train_list, label_train_list, img_test_list, label_test_list = split(img_list, label_list)
    train = pd.DataFrame()
    test = pd.DataFrame()
    train['img'] = img_train_list
    train['label'] = label_train_list
    test['img'] = img_test_list
    test['label'] = label_test_list
    train.to_csv("D:/study/iris/csv/train.csv", index=False, encoding="utf-8")
    test.to_csv("D:/study/iris/csv/test.csv", index=False, encoding="utf-8")
    print("finish!")

def main1():
    file_dir = "D:/study/iris/data"
    img_train_list, label_train_list, img_test_list, label_test_list = get_files1(file_dir)
    train = pd.DataFrame()
    test = pd.DataFrame()
    train['img'] = img_train_list
    train['label'] = label_train_list
    test['img'] = img_test_list
    test['label'] = label_test_list
    train.to_csv("D:/study/iris/csv/train1.csv", index=False, encoding="utf-8")
    test.to_csv("D:/study/iris/csv/test1.csv", index=False, encoding="utf-8")
    print("finish!")
if __name__=="__main__":
    main1()