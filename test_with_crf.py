import numpy as np
import scipy.misc
import tensorflow as tf
import os
import time
import sys
sys.path.append("~/tensorflow_models/slim")
from segMnet import segMnet
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import * 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
multiplier=1.0

MODEL_SAVE_PATH='./model/'
HEIGHT=256
WIDTH=192

def test(filename, save_name):
    MEAN_VALUES = np.array([123.68, 116.78, 103.94])
    MEAN_VALUES = MEAN_VALUES.reshape((1, 1, 1, 3))
    image = scipy.misc.imread(filename, mode='RGB')
    image = scipy.misc.imresize(image, (HEIGHT,WIDTH))
    h,w,d = img.shape
    timg = np.reshape(image, (1, h, w, 3)) - MEAN_VALUES
    
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, shape=[1, h, w, d])
        genered = tf.nn.softmax(tf.squeeze(segMnet(images, multiplier),axis=0))
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
    unary = unary_from_softmax(result.transpose((2,0,1)))
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF(h*w, 2)
    d.setUnaryEnergy(unary)
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20), img=image, chdim=2)
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(50)
    MAP = np.argmax(Q,axis=0).reshape(h,w)

    img = np.zeros((h,w,4), dtype=np.int)
    img[:,:,0:3] = image
    img[:,:,3] = MAP * 255
    scipy.misc.imsave(save_name, img)

if __name__ == '__main__':
    test('test.jpg', 'result.png')
