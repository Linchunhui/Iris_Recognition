import tensorflow as tf
import numpy as np
import pandas as pd
from ResNet import resnet_base,get_restorer
from inceptionv4 import inception_v4,inception_v4_base
from DenseNet import densenet121
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow
import cv2
import os

def init_tf():
    global sess, logits, x
    # process image
    x = tf.placeholder(tf.float32, shape=[64,512, 3])
    x_4d = tf.reshape(x, [-1, 64, 512, 3])
    logits = resnet_base(x_4d,"resnet_v1_101",is_training=False)
    logits= tf.reshape(logits,[-1])
    print("logit", np.shape(logits))
    restorer, checkpoint_path=get_restorer()
    sess = tf.Session()
    restorer.restore(sess, checkpoint_path)
    print('load model done...')

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print([str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def init_tf_dense():
    global sess, logits, x
    # process image
    checkpoint_path = r"D:\study\iris\code\CNN\model\tf-densenet121.ckpt"
    x = tf.placeholder(tf.float32, shape=[64, 512, 3])
    x_4d = tf.reshape(x, [-1, 64, 512, 3])
    logits = densenet121(x_4d, num_classes=1000, is_training=False)
    print("logit1", np.shape(logits))
    logits= tf.reshape(logits, [-1])
    print("logit", np.shape(logits))
    restore_variable = slim.get_variables_to_restore()
    print(restore_variable)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, restore_variable, ignore_missing_vars=True)
    sess = tf.Session()
    init_fn(sess)
    initialize_uninitialized(sess)
    print('load model done...')

def inceptionv4_init():
    global sess, logits, x
    # process image
    checkpoint_path = r"D:\study\iris\code\CNN\model\inception_v4.ckpt"
    x = tf.placeholder(tf.float32, shape=[64, 512, 3])
    x_4d = tf.reshape(x, [-1, 64, 512, 3])
    logits,_ = inception_v4_base(x_4d)
    logits = tf.reshape(logits, [-1])
    print("logit", np.shape(logits))
    restore_variable = slim.get_variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, restore_variable, ignore_missing_vars=True)
    sess = tf.Session()
    init_fn(sess)
    initialize_uninitialized(sess)

def feature_ex(img_path):
    #img = image.load_img(img_path, target_size=(size, size))
    img = cv2.imread(img_path,0)
    #img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_array = np.array(img)
    feature = sess.run(logits, feed_dict={x:img_array})
    return feature

def ex_train():
    train=pd.read_csv("D:/study/iris/csv/train.csv")
    dir = r"D:\study\iris\CASIA\norm_512\train"
    train_img_list = train["img"]
    train_label_list = train["label"]
    train_size = len(train_img_list)
    feature_list=[]
    for i in range(train_size):
        path = os.path.join(dir,train_img_list[i].split("\\")[-1])
        feature = np.array(feature_ex(path)).astype(np.float16)
        feature_list.append(feature)
        print("train:", i, path)
    return np.array(feature_list)
    #cnn_feature = pd.DataFrame(np.array(feature_list))
    #cnn_feature.to_csv("D:/study/iris/csv/train_vector2.csv", index=False, encoding="utf-8")

def ex_test():
    test=pd.read_csv("D:/study/iris/csv/test.csv")
    dir = r"D:\study\iris\CASIA\norm_512\test"
    test_img_list = test["img"]
    test_label_list = test["label"]
    test_size = len(test_img_list)
    feature_list=[]
    for i in range(test_size):
        path = os.path.join(dir,test_img_list[i].split("\\")[-1])
        feature = np.array(feature_ex(path)).astype(np.float16)
        feature_list.append(feature)
        print("test:", i, path)
    return np.array(feature_list)
    #cnn_feature = pd.DataFrame(np.array(feature_list))
    #cnn_feature.to_csv("D:/study/iris/csv/test_vector2.csv", index=False, encoding="utf-8")


if __name__=="__main__":
    init_tf_dense()
    #ex_train()
    #ex_test()