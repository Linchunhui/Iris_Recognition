import pandas as pd
import shutil
import os

train = pd.read_csv("D:/study/iris/csv/train.csv")
test = pd.read_csv("D:/study/iris/csv/test.csv")

train_img_list = train["img"]
train_label_list = train["label"]
test_img_list = test["img"]
test_label_list = test["label"]
train_size = len(train_img_list)
test_size = len(test_img_list)
copy_dir1="D:/study/iris/CASIA/origin/train"
copy_dir2="D:/study/iris/CASIA/origin/test"

for i in range(train_size):
    train_path=train_img_list[i]
    #print(train_path)
    img_name=train_path.split('\\')[-1]
    copy_path1=os.path.join(copy_dir1,img_name)
    print(train_path,copy_path1)
    shutil.copy(train_path,copy_path1)
for i in range(test_size):
    test_path = test_img_list[i]
        # print(train_path)
    img_name = test_path.split('\\')[-1]
    copy_path2 = os.path.join(copy_dir2, img_name)
    shutil.copy(test_path,copy_path2)

    '''img = cv2.imread(train_path, 0)
    iris, pupil = IrisLocalization(img)
    #cv2.circle(img, (iris[0], iris[1]), iris[2], (0, 0, 255), 1)
    #cv2.circle(img, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 1)
    #cv2.imwrite("D:/study/iris/process_data/circle/train/{}.jpg".format(img_name),img)
    normalized = IrisNormalization(img, pupil, iris)
    cv2.imwrite("D:/study/iris/process_data/rectangle/train/{}.jpg".format(img_name),normalized)
    ROI = ImageEnhancement(normalized)
    cv2.imwrite("D:/study/iris/process_data/enhancement/train/{}.jpg".format(img_name), ROI)
    #train_features[i, :] = FeatureExtraction(ROI)
    #train_classes[i] = train_label_list[i]
    print('train_feature:',train_img_list[i],train_label_list[i],'{}/{}'.format(i,train_size))'''