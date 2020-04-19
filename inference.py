import os
import cv2
import time

import numpy as np
import tensorflow as tf

from absl import app, flags, logging

from PWCDCNet import PWCDCNet
from flow_utils import flow_to_image, read_flow

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_path', './checkpoints/ckpt-1200000', 'Link to the file of TensorFlow checkpoint.')
flags.DEFINE_string('image_0', './sample_images/00003_img1.ppm', None)
flags.DEFINE_string('image_1', './sample_images/00003_img2.ppm', None)
flags.DEFINE_string('output_dir', './sample_images/', None)

def restore(net, ckpt_path):
    checkpoint = tf.train.Checkpoint(net=net)
    if os.path.isdir(ckpt_path):
        latest_checkpoint = tf.train.latest_checkpoint(ckpt_path)

        status = checkpoint.restore(latest_checkpoint).expect_partial()

        logging.info("Restored from {}".format(latest_checkpoint))

    elif os.path.exists('{}.index'.format(ckpt_path)):
        status = checkpoint.restore(ckpt_path).expect_partial()

        logging.info("Restored from {}".format(ckpt_path))

    else:
        logging.info("Nothing to restore.")

@tf.function
def inference(image_pairs, model):
    _, h, w, _ = tf.unstack(tf.shape(image_pairs))

    # Check if the shape of image can be divided by 64, as we have a 6-level feature extractor.
    if h % 64 != 0 or w % 64 != 0:
        new_h = (int(h/64) + 1) * 64
        new_w = (int(w/64) + 1) * 64

        image_pairs = tf.image.pad_to_bounding_box(image_pairs, 0, 0, new_h, new_w)

    flo_pred = model(image_pairs, is_training=False)
    flo_pred = tf.image.crop_to_bounding_box(flo_pred, 0, 0, h//4, w//4)

    flo_pred = tf.image.resize(flo_pred, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    flo_pred *= 4

    return flo_pred[0]

def main(argv):
    ''' Initialize model '''
    pwcnet = PWCDCNet()

    restore(net=pwcnet, ckpt_path=FLAGS.ckpt_path)

    image_0 = cv2.imread(FLAGS.image_0) 
    image_1 = cv2.imread(FLAGS.image_1)

    image_0 = np.array(image_0, dtype=np.float32) / 255.0
    image_1 = np.array(image_1, dtype=np.float32) / 255.0

    image_pairs = np.concatenate([image_0, image_1], axis=2)
    image_pairs = np.expand_dims(image_pairs, axis=0)
   
    flo_pred = inference(image_pairs, model=pwcnet)

    flo_im = flow_to_image(flo_pred.numpy())
    flo_im = cv2.cvtColor(flo_im, cv2.COLOR_BGR2RGB)
    
    fn = FLAGS.image_0.split('/')[-1].split('.')[0]
    cv2.imwrite('sample_images/{}_output.png'.format(fn), flo_im)

    logging.info('Save output to "sample_images/{}_output.png"'.format(fn))
    
if __name__ == '__main__':
    app.run(main)