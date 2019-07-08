import tensorflow as tf
import pandas as pd
def get_train_list():
    #dir="D:/study/iris/process_data/rectangle/train/"
    train = pd.read_csv("D:/study/iris/csv/train.csv")
    img_list = train['img'].apply(lambda x:"D:/study/iris/CASIA/norm_512/train/"+x.split('\\')[-1].split('.')[0]+".jpg")
    print(img_list)
    label_list = train['label']
    return img_list, label_list

def get_test_list():
    #dir="D:/study/iris/process_data/rectangle/train/"
    test = pd.read_csv("D:/study/iris/csv/test.csv")
    img_list = test['img'].apply(lambda x:"D:/study/iris/CASIA/norm_512/test/"+x.split('\\')[-1].split('.')[0]+".jpg")
    label_list = test['label']
    return img_list, label_list

def get_batch(image, label, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)
    image = tf.image.resize_images(image,(64,64))

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    #tf.summary.image("input_img", image_batch, max_outputs=5)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
if __name__=="__main__":
    get_train_list()