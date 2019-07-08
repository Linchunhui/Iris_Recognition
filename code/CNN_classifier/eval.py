import numpy as np
import os
from utils import *
from DenseNet import densenet
import cv2
#net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设置最小gpu使用量

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

log_train_dir="D:/study/iris/code/CNN/log/train6/model.ckpt-43749"

def main():
    init_tf(log_train_dir, 1000)
    func1()

def init_tf(logs_train_dir,N_CLASSES):
    global sess, pred, x
    # process image
    x = tf.placeholder(tf.float32, shape=[64, 64])
    #x_norm = tf.image.per_image_standardization(x)
    x_4d = tf.reshape(x, [-1, 64, 64, 1])
    # predict
    logits = densenet(x_4d,1.0,N_CLASSES)
    print("logit", np.shape(logits))
    pred = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, logs_train_dir)
    print('load model done...')


def evaluate_image2(img_path,size):
    #img = image.load_img(img_path, target_size=(size, size))
    image = cv2.imread(img_path)
    img = image[:, :, 0]
    img = cv2.resize(img, (size,size))
    image_array = np.array(img).reshape(size,size)
#    image_array = image.img_to_array(img)
    prediction = sess.run(pred, feed_dict={x: image_array})
    return prediction

def func1(size=64):
    count=0
    test_img, test_label=get_test_list()
    for i in range(len(test_img)):
        pred = evaluate_image2(test_img[i], size)
        index = np.argmax(pred)
        print(test_img[i],index,test_label[i])
        if index+1==test_label[i]:
            count+=1
    #print(count)
    print((count/len(test_img))*1.00)
if __name__ == "__main__":
    main()