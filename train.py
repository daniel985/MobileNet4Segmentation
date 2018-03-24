import os
import scipy.misc
import random
import sys
sys.path.append("~/tensorflow_models/slim")
import numpy as np
import tensorflow as tf
from segMnet import segMnet

slim = tf.contrib.slim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
checkpoint_dir = './mnets/'
checkpoint_path = os.path.join(checkpoint_dir, 'mobilenet_v1_1.0_224.ckpt')
multiplier=1.0

TRAIN_IMAGE_DIRECTORY = './data/'
MODEL_SAVE_PATH='./model/'
BATCH_SIZE = 8
NUM_EPOCHS = 500
HEIGHT=256
WIDTH=256

def preprocess(image,height=HEIGHT,width=WIDTH):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    MEAN_VALUES = np.array([123.68, 116.78, 103.94])
    img = image[:,:,:3] - MEAN_VALUES
    img.set_shape(shape=(height,width,3))
    mask = tf.cast(image[:,:,:,3], tf.uint8)
    mask = tf.expand_dims(mask, axis=2)
    mask.set_shape(shape=(height,width,1))
    return img, mask

def train():

    with tf.Session() as sess:
        all_images = os.listdir(TRAIN_IMAGE_DIRECTORY)
        image_files = [os.path.join(TRAIN_IMAGE_DIRECTORY, image) for image in all_images]
        num_steps_per_epoch = len(image_files) / BATCH_SIZE
        images = tf.convert_to_tensor(image_files)
        input_queue = tf.train.slice_input_producer([images], num_epochs=NUM_EPOCHS, seed=31)
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_png(image, channels=4)
        preprocessed_image, preprocessed_label = preprocess(image)
        images, labels = tf.train.batch([preprocessed_image, preprocessed_label], batch_size=BATCH_SIZE, allow_smaller_final_batch=True)

        logits = segMnet(images, multiplier)
        labels = tf.reshape(labels, [-1])
        labels_oht = tf.one_hot(labels, 2)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_oht, logits=tf.reshape(logits, [-1, 2])))
        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(
                learning_rate = 1e-2,
                global_step = global_step,
                decay_steps = 10*num_steps_per_epoch,
                decay_rate = 0.5,
                staircase = True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(cost, global_step=global_step)
        
        mnet_weights = slim.get_model_variables('MobilenetV1')
        read_mnet_weight_func = slim.assign_from_checkpoint_fn(checkpoint_path, mnet_weights)
        store_variables = []
        for v in tf.global_variables():
            if not 'Adam' in v.name:
                store_variables.append(v)
        saver = tf.train.Saver(store_variables)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        
        model_file = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
        if model_file:
            print ("restore from %s"%(model_file))
            saver.restore(sess, model_file)
        else:
            print ("init from MobilenetV1")
            read_mnet_weight_func(sess)

        print('train start:')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                _, loss, step = sess.run([train_op, cost, global_step])
                print(step, loss)
                if step % 10000 == 0:
                    saver.save(sess, MODEL_SAVE_PATH, global_step=step)
                    print('Saved, step %d' % step)
        except:
            saver.save(sess, MODEL_SAVE_PATH + "-done")
            print ("train finished!")
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()
