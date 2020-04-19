import os
import time

import numpy as np
import tensorflow as tf

from absl import app, flags, logging

from PWCDCNet import PWCDCNet
from data_loader import DataLoader
from flow_utils import flow_to_image

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/work/flow_dataset', 'Link to dataset directory.')
flags.DEFINE_string('val_list', './lists/FlyingChairs_val_list.txt', 'Link to validation list.')
flags.DEFINE_string('save_dir', './checkpoints', 'Link to checkpoints directory.')

flags.DEFINE_list('losses_weight', [0.32, 0.08, 0.02, 0.01, 0.005], 'Loss weights for 6th to 2nd flow predictions, as described in the original paper.')
flags.DEFINE_float('gamma', 0.0004, None)
flags.DEFINE_integer('batch_size', 8, None)
flags.DEFINE_enum('dataset', 'mixed', ['mixed', 'chairs', 'things3d_ft'], None)

# Learing schedule for training from scratch. Train on the mixed dataset of FlyingChairs and FlyingThings3D.
flags.DEFINE_string('train_list', './lists/FlyingChairs-Things3D-Mixed_train_list.txt', 'Link to training list.')
flags.DEFINE_list('lr_boundaries', [400000, 600000, 800000, 1000000], None)
flags.DEFINE_float('lr', 0.0001, None)
flags.DEFINE_integer('num_steps', 1500000, None)
flags.DEFINE_list('crop_size', [256, 448], None)

flags.DEFINE_integer('steps_per_save', 10000, None)
flags.DEFINE_integer('steps_per_eval', 1000, None)
flags.DEFINE_integer('log_freq', 50, None)

flags.DEFINE_boolean('random_scale', False, 'Random scale.')
flags.DEFINE_boolean('random_flip', False, 'Random flip.')

# Can converge w/ or w/o @tf.function, however, it is really slow w/o using @tf.function: ~0.6sec/step. (updated at 2020/04/07)
@tf.function
def train_step(model, image_pairs, flo_gt, metric, summary_writer):
    logging.info('Tracing, in Func. "train_step" ...')

    flo_preds = model(image_pairs) 

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
    l2_losses = [FLAGS.gamma * tf.nn.l2_loss(v) for v in model.trainable_variables]
    l2_losses = tf.reduce_sum(l2_losses)

    total_losses = flo_losses + l2_losses
    
    return total_losses, flo_preds

@tf.function
def eval_step(model, image_pairs, flo_gt):
    flo_preds = model(image_pairs, is_training=False)
    EPE = end_point_error(flo_preds=flo_preds, flo_gt=flo_gt)

    return EPE 

@tf.function
def end_point_error(flo_preds, flo_gt):
    logging.info('Tracing, in Func. "end_point_error" ...')
 
    _, gt_height, _, _ = tf.unstack(tf.shape(flo_gt))
    _, pred_height, _, _ = tf.unstack(tf.shape(flo_preds))

    flo_preds = tf.image.resize(flo_preds, tf.shape(flo_gt)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    flo_scale = tf.cast(gt_height / pred_height, dtype=tf.float32)
    flo_preds *= flo_scale

    error = tf.reduce_mean(tf.norm(flo_preds-flo_gt, ord='euclidean', axis=3))

    return error

def write_summary(summary_writer, step, metric, mode, image_pairs=None, flo_preds=None, flo_gt=None):
    with summary_writer.as_default():  
        if mode == 'training':
            tf.summary.image('image_1', image_pairs[:, :, :, :3], step=step, max_outputs=3)
            tf.summary.image('image_2', image_pairs[:, :, :, 3:], step=step, max_outputs=3)

            flo_gt_colored = np.stack([flow_to_image(f) for f in flo_gt.numpy()], axis=0)
            tf.summary.image('flow_gt', flo_gt_colored, step=step, max_outputs=3)

            # Summary each pyramid level 
            for i, flo_pred in enumerate(flo_preds):
                flo_pred_colored = np.stack([flow_to_image(f) for f in flo_pred.numpy()], axis=0)

                tf.summary.image('flow_pred_{}'.format(i), flo_pred_colored, step=step, max_outputs=3)

            tf.summary.scalar('training_loss', metric.result(), step=step)

        elif mode == 'validation':
            tf.summary.scalar('val_AEPE', metric.result(), step=step)

def main(argv):
    ''' Prepare dataset '''
    data_loader = DataLoader(FLAGS.data_dir, FLAGS.train_list, FLAGS.val_list)
    train_dataset, val_dataset = data_loader.create_tf_dataset(flags=FLAGS)

    ''' Declare and setup optimizer '''
    num_steps = FLAGS.num_steps // (FLAGS.batch_size // 8) + 1
    lr_boundaries = [x // (FLAGS.batch_size // 8) for x in FLAGS.lr_boundaries] # Adjust the boundaries by batch size
    lr_values = [FLAGS.lr/(2**i) for i in range(len(FLAGS.lr_boundaries)+1)] 
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries, values=lr_values)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    
    logging.info('Learning rate boundaries: {}'.format(lr_boundaries))
    logging.info('Training steps: {}'.format(num_steps))

    ''' Create metric and summary writers '''
    train_metric = tf.keras.metrics.Mean(name='train_loss')
    val_metric = tf.keras.metrics.Mean(name='val_average_end_point_error')
    time_metric = tf.keras.metrics.Mean(name='elapsed_time_per_step')

    train_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.save_dir, 'summaries', 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.save_dir, 'summaries', 'val'))

    ''' Initialize model '''
    pwcnet = PWCDCNet()
    
    ''' Check if there exists the checkpoints '''
    ckpt_path = os.path.join(FLAGS.save_dir, 'tf_ckpt')
    ckpt = tf.train.Checkpoint(optimizer=optimizer, net=pwcnet)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=20)

    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    status = ckpt.restore(manager.latest_checkpoint).expect_partial()
    
    ''' Start training '''
    step = optimizer.iterations.numpy()
    while step < num_steps:
        for im_pairs, flo_gt in train_dataset:
            # Model inference and use 'tf.GradientTape()' to trace gradients.
            start_time = time.time()
            with tf.GradientTape() as tape:
                total_losses, flo_preds = train_step(model=pwcnet,
                                        metric=train_metric, summary_writer=train_summary_writer,
                                        image_pairs=im_pairs, flo_gt=flo_gt)

            # Update weights. Compute gradients and apply to the optimizersr.
            grads = tape.gradient(total_losses, pwcnet.trainable_variables)
            optimizer.apply_gradients(zip(grads, pwcnet.trainable_variables))

            elapsed_time = time.time() - start_time

            # Logging
            train_metric.update_state(total_losses)
            time_metric.update_state(elapsed_time)

            step = optimizer.iterations.numpy()
            if step % FLAGS.log_freq == 0:
                write_summary(summary_writer=train_summary_writer, step=step,
                                metric=train_metric, mode='training',
                                image_pairs=im_pairs, flo_preds=flo_preds, flo_gt=flo_gt)
                
                logging.info('Step {:>7}, Training Loss: {:.5f}, ({:.3f} sec/step)'.format(step,
                                                                                    train_metric.result(), 
                                                                                    time_metric.result()))
                train_metric.reset_states()
                time_metric.reset_states()

            # Evaluate 
            if step % FLAGS.steps_per_eval == 0:
                for im_pairs, flo_gt in val_dataset:
                    EPE = eval_step(model=pwcnet, image_pairs=im_pairs, flo_gt=flo_gt)
                    val_metric.update_state(EPE)

                write_summary(summary_writer=val_summary_writer, step=step,
                                metric=val_metric, mode='validation')

                logging.info('*****Steps {:>7}, AEPE = {:.5f}*****'.format(step, val_metric.result()))

                val_metric.reset_states()

            # Save checkpoints
            if step % FLAGS.steps_per_save == 0:
                manager.save(checkpoint_number=step)
                logging.info('*****Steps {:>7}, save checkpoints!*****'.format(step))

if __name__ == '__main__':
    app.run(main)