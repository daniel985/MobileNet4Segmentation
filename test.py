import os
import time
import scipy.misc
import sys
sys.path.append("~/tensorflow_models/slim")
import numpy as np
from segMnet2 import segMnet
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
multiplier=1.0

MODEL_SAVE_PATH='./model/'
HEIGHT=256
WIDTH=256

def test(filename, save_name):
    MEAN_VALUES = np.array([123.68, 116.78, 103.94])
    MEAN_VALUES = MEAN_VALUES.reshape((1, 1, 1, 3))
    image = scipy.misc.imread(filename, mode='RGB')
    image = scipy.misc.imresize(image, (HEIGHT,WIDTH))
    h,w,d = image.shape
    timg = np.reshape(image, (1, h, w, 3)) - MEAN_VALUES

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, shape=[1, h, w, 3])
        genered = tf.argmax(segMnet(images, multiplier), dimension=3)
        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
        if model_file:
            saver.restore(sess, model_file)
        else:
            raise Exception('Testing needs pre-trained model!')

        feed_dict = {images : timg}
        start = time.time()
        result = sess.run(genered,feed_dict=feed_dict)
        end = time.time()
        print ("cost time:%f"%(end-start))
    
    img = np.zeros((h,w,4), dtype=np.int)
    img[:,:,0:3] = image
    img[:,:,3] = result * 255
    scipy.misc.imsave(save_name, img)

if __name__ == '__main__':
    test('test.jpg', 'result.png')
