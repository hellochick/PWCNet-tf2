import os
import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from absl import app, flags, logging

from PWCDCNet import PWCDCNet
from data_loader import DataLoader
from flow_utils import flow_to_image

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/work/flow_dataset', 'Link to dataset directory.')
flags.DEFINE_string('ckpt_path', './checkpoints/ckpt-1200000', 'Link to the directory/file of TensorFlow checkpoint.')
flags.DEFINE_string('val_list', './lists/FlyingChairs_val_list.txt', 'Link to validation list.')

@tf.function
def eval_step(model, image_pairs, flo_gt):
    logging.info('Tracing, in Func. "eval_step" ...')
    
    _, h, w, _ = tf.unstack(tf.shape(image_pairs))

    # Check if the shape of image can be divided by 64, as we have a 6-level feature extractor.
    if h % 64 != 0 or w % 64 != 0:
        new_h = (int(h/64) + 1) * 64
        new_w = (int(w/64) + 1) * 64
        shape = [new_h, new_w]
        
        image_pairs = tf.image.pad_to_bounding_box(image_pairs, 0, 0, new_h, new_w)

    flo_pred = model(image_pairs, is_training=False)
    flo_pred = tf.image.crop_to_bounding_box(flo_pred, 0, 0, h//4, w//4)

    EPE = end_point_error(flo_pred=flo_pred, flo_gt=flo_gt)

    return EPE 

@tf.function
def end_point_error(flo_pred, flo_gt):
    logging.info('Tracing, in Func. "end_point_error" ...')
 
    _, gt_height, _, _ = tf.unstack(tf.shape(flo_gt))
    _, pred_height, _, _ = tf.unstack(tf.shape(flo_pred))

    flo_pred = tf.image.resize(flo_pred, tf.shape(flo_gt)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    flo_scale = tf.cast(gt_height / pred_height, dtype=tf.float32)
    flo_pred *= flo_scale

    error = tf.reduce_mean(tf.norm(flo_pred-flo_gt, ord='euclidean', axis=3))

    return error

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

def main(argv):
    ''' Prepare dataset '''
    data_loader = DataLoader(FLAGS.data_dir, val_list=FLAGS.val_list)
    _, val_dataset = data_loader.create_tf_dataset(flags=FLAGS)
    logging.info('Number of validation samples: {}'.format(data_loader.val_size))

    ''' Create metric and summary writers '''
    val_metric = tf.keras.metrics.Mean(name='val_average_end_point_error')

    ''' Initialize model '''
    pwcnet = PWCDCNet()

    restore(net=pwcnet, ckpt_path=FLAGS.ckpt_path)

    with tqdm(total=data_loader.val_size) as pbar:
        pbar.set_description('Evaluation progress: ')

        for im_pairs, flo_gt in val_dataset:
            EPE = eval_step(model=pwcnet, image_pairs=im_pairs, flo_gt=flo_gt)
            val_metric.update_state(EPE)

            pbar.update(1)

    logging.info('*****AEPE = {:.5f}*****'.format(val_metric.result()))
    val_metric.reset_states()

    ''' 
    For those people want to choose the best model for validation data, this part 
    can be used to run over all the checkpoints in the directory.
    '''

    # checkpoint = tf.train.Checkpoint(net=pwcnet)
    # manager = tf.train.CheckpointManager(checkpoint, FLAGS.ckpt_path, max_to_keep=20)

    # for ckpt_path in manager.checkpoints:
        # status = checkpoint.restore(ckpt_path).expect_partial()
        # logging.info("Restored from {}".format(ckpt_path))

        # with tqdm(total=data_loader.val_size) as pbar:
        #     pbar.set_description('Evaluation progress: ')

        #     for im_pairs, flo_gt in val_dataset:
        #         EPE = eval_step(model=pwcnet, image_pairs=im_pairs, flo_gt=flo_gt)
        #         val_metric.update_state(EPE)

        #         pbar.update(1)

        # logging.info('*****AEPE = {:.5f}*****'.format(val_metric.result()))
        # val_metric.reset_states()

if __name__ == '__main__':
    app.run(main)