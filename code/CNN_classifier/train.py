# coding:utf-8
import os
import tensorflow as tf
from utils import *
from DenseNet import densenet
import tensorflow.contrib.slim as slim

keep_prob = 0.5
batch_size = 32
num_class = 1000
init_lr = 0.01
decay_steps = 30
epochs = 100
log_dir="D:/study/iris/code/CNN/log/train6"

def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # Convolution Layer with 24 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 24, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 48 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 48, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 768)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out

def main():
    train(BATCH_SIZE=batch_size,N_CLASSES=num_class,init_lr= init_lr,\
          decay_steps=decay_steps,logs_train_dir=log_dir,epochs=epochs)


CAPACITY = 1000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设置最小gpu使用量

def train(BATCH_SIZE,N_CLASSES,init_lr,decay_steps,logs_train_dir,epochs):
    train_list,train_label=get_train_list()
    one_epoch_step = len(train_list) / BATCH_SIZE
    MAX_STEP=epochs * one_epoch_step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # label without one-hot
    batch_train, batch_labels = get_batch(train_list, train_label, BATCH_SIZE, CAPACITY)
    logits =densenet(batch_train, keep_prob, N_CLASSES, True)
    #net
    print(logits.get_shape())
    # loss
    label_smoothing = 0.1
    one_hot_labels = slim.one_hot_encoding(batch_labels, N_CLASSES)
    one_hot_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / N_CLASSES #标签平滑

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('train_loss', loss)
    # optimizer
    lr = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step, decay_steps=decay_steps*one_epoch_step,
                                    decay_rate=0.1)
    tf.summary.scalar('learning_rate', lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.cast(batch_labels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('train_acc', accuracy)

    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=config)
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    # saver = tf.train.Saver()
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)

    #saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #saver.restore(sess, logs_train_dir+'/model.ckpt-10000')
    try:
        for step in range(int(MAX_STEP)):
            if coord.should_stop():
                break
            _, learning_rate, tra_loss, tra_acc = sess.run([optimizer, lr, loss, accuracy])
            if step % 10 == 0:
                print('Epoch %3d/%d, Step %6d/%d, lr %f, train loss = %.2f, train accuracy = %.2f%%' % (
                step / one_epoch_step, MAX_STEP / one_epoch_step, step, MAX_STEP, learning_rate, tra_loss,
                tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 5000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()