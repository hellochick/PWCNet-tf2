import os
import cv2
import time
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from absl import app, flags, logging

from flow_utils import flow_to_image

from data_loader_v1_compat import DataLoader
from PWCDCNet_v1_compat import PWCDCNet

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/work/flow_dataset', 'Link to dataset directory.')
flags.DEFINE_string('train_list', './lists/FlyingChairs-Things3D-Mixed_train_list.txt', 'Link to training list.')
flags.DEFINE_string('val_list', './lists/FlyingChairs_val_list.txt', 'Link to validation list.')
flags.DEFINE_string('model_dir', './checkpoints', 'Link to checkpoints directory.')
flags.DEFINE_enum('dataset', 'mixed', ['mixed', 'chairs', 'things3d_ft'], None)

flags.DEFINE_list('losses_weight', [0.32, 0.08, 0.02, 0.01, 0.005], 'Loss weights for 6th to 2nd flow predictions, as described in the original paper.')
flags.DEFINE_float('gamma', 0.0004, None)
flags.DEFINE_list('lr_boundaries', [400000, 600000, 800000, 1000000], None)
flags.DEFINE_float('lr', 0.0001, None)
flags.DEFINE_integer('batch_size', 8, None)
flags.DEFINE_list('crop_size', [256, 448], None)
flags.DEFINE_integer('n_levels', 6, None)
flags.DEFINE_integer('output_level', 2, None)

flags.DEFINE_integer('num_steps', 1500000, None)
flags.DEFINE_integer('steps_per_save', 1000, None)
flags.DEFINE_integer('steps_per_eval', 1000, None)
flags.DEFINE_integer('log_freq', 50, None)

flags.DEFINE_boolean('random_scale', False, 'Random scale.')
flags.DEFINE_boolean('random_flip', False, 'Random flip.')

def loss_fn(flo_preds, flo_gt):
    # Use multi-scale loss, as described in Sec. 3 in the original paper.
    flo_losses = 0.
    for flo_pred, weight in zip(flo_preds, FLAGS.losses_weight):
        _, gt_height, _, _ = tf.unstack(tf.shape(flo_gt))
        _, pred_height, _, _ = tf.unstack(tf.shape(flo_pred))

        scaled_flow_gt = tf.image.resize(flo_gt, tf.shape(flo_pred)[1:3], method=tf.image.ResizeMethod.BILINEAR)
        scaled_flow_gt /= tf.cast(gt_height / pred_height, dtype=tf.float32)

        l2_norm = tf.norm(flo_pred-scaled_flow_gt, ord=2, axis=3)
        flo_loss = tf.reduce_mean(tf.reduce_sum(l2_norm, axis=(1, 2)))

        flo_losses += flo_loss * weight

    # Calculate the L2 norm to regularize. 
    l2_losses = [FLAGS.gamma * tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    l2_losses = tf.reduce_sum(l2_losses)

    total_losses = flo_losses + l2_losses
    
    return total_losses

def end_point_error_fn(flo_preds, flo_gt):
    _, gt_height, _, _ = tf.unstack(tf.shape(flo_gt))
    _, pred_height, _, _ = tf.unstack(tf.shape(flo_preds))

    flo_preds = tf.image.resize(flo_preds, tf.shape(flo_gt)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    flo_scale = tf.cast(gt_height / pred_height, dtype=tf.float32)
    flo_preds *= flo_scale

    error = tf.reduce_mean(tf.norm(flo_preds-flo_gt, ord='euclidean', axis=3))

    return error

def main(argv):
    ''' Prepare dataset and make dataset iterator '''
    data_loader = DataLoader(FLAGS.data_dir, FLAGS.train_list, FLAGS.val_list)
    train_dataset, val_dataset = data_loader.create_tf_dataset(flags=FLAGS)

    im_pairs_train, flow_gt_train = train_dataset.make_one_shot_iterator().get_next() 
    im_pairs_val, flow_gt_val = val_dataset.make_one_shot_iterator().get_next() 

    ''' Setup model for training and validation'''
    pwcnet_train = PWCDCNet(FLAGS, im_pairs_train)
    with tf.variable_scope('', reuse=True):
        pwcnet_val = PWCDCNet(FLAGS, im_pairs_val)

    ''' Calculate the losses '''
    total_losses = loss_fn(flo_preds=pwcnet_train.flow_outputs, flo_gt=flow_gt_train)
    
    ''' Setup learning rate scheduler '''
    lr_boundaries = [x // (FLAGS.batch_size // 8) for x in FLAGS.lr_boundaries] # Adjust the boundaries by batch size
    lr_values = [FLAGS.lr/(2**i) for i in range(len(FLAGS.lr_boundaries)+1)] 

    global_step = tf.train.get_or_create_global_step()
    update_op = tf.assign_add(global_step, 1)

    with tf.control_dependencies([update_op]):
        lr_value = tf.train.piecewise_constant(global_step, boundaries=lr_boundaries, values=lr_values, name='learning_rate')

        opt_conv = tf.train.AdamOptimizer(lr_value)
        grads = tf.gradients(total_losses, tf.trainable_variables())
        train_op = opt_conv.apply_gradients(zip(grads, tf.trainable_variables()))

    ''' Monitor the end point error of validation data. '''
    epe_val = end_point_error_fn(flo_preds=pwcnet_val.flow_outputs[-1], flo_gt=flow_gt_val)

    ''' Setup TensorFlow Config for training, create session, and initialize variables '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    sess = tf.Session(config=config)
    sess.run([global_init, local_init])

    ''' Start training '''
    losses, times = [], []
    for step in range(FLAGS.num_steps):
        start_time = time.time()

        # _loss, _flow_preds, _lr, _global_step, _ = sess.run([total_losses, pwcnet_train.flow_outputs[-1], lr_value, global_step, train_op])
        _loss, _lr, _global_step, _ = sess.run([total_losses, lr_value, global_step, train_op])

        losses.append(_loss)
        times.append(time.time()-start_time)

        if step % FLAGS.log_freq == 0:
            logging.info('Step {:>7}, Learning Rate: {:>6f}, Training Loss: {:.5f}, ({:.3f} sec/step),'.format(step, _lr,
                                                                                                            np.mean(losses),
                                                                                                            np.mean(times)))
            losses, times = [], []

        if step % FLAGS.steps_per_eval == 0:
            val_epes = []
            for i in range(data_loader.val_size):
                _epe, _flow_preds = sess.run([epe_val, pwcnet_val.flow_outputs[-1]])

                val_epes.append(_epe)

            logging.info('*****Steps {:>7}, Validation AEPE = {:.5f}*****'.format(step, np.mean(val_epes)))


if __name__ == '__main__':
    app.run(main)