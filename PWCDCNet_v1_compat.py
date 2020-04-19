import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from functools import partial
from flow_utils import bilinear_warp

tf.disable_v2_behavior()
print('TensorFlow Version: ', tf.__version__)

class PWCDCNet(object):
    def __init__(self, flags, im_pairs_tensor):
        # flow_scales for level from 0 -> 6
        self.flow_scales = [20.0, 10.0, 5.0, 2.5, 1.25, 0.625, None]
        self.output_level = flags.output_level
        self.n_levels = flags.n_levels

        self.flow_outputs = self.build_graph(im_pairs_tensor)
        
    def feature_extractor(self, inputs, num_filters=[16, 32, 64, 96, 128, 196]):
        x = inputs 

        ft_pyramids = []
        for i, n in enumerate(num_filters):
            x = tf.layers.Conv2D(filters=n, kernel_size=3, strides=2, kernel_initializer='he_normal', padding='same', name='conv_{}_1'.format(i+1))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(filters=n, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', name='conv_{}_2'.format(i+1))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(filters=n, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', name='conv_{}_3'.format(i+1))(x)
            x = tf.nn.leaky_relu(x, 0.1)

            ft_pyramids.append(x)

        return ft_pyramids[::-1]

    def residual_block(self, inputs, num_filters, level):
        features = inputs
        for i, n in enumerate(num_filters):
            x = tf.layers.Conv2D(filters=n, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', name='conv_{}_{}'.format(level, i+1))(features)
            x = tf.nn.leaky_relu(x, 0.1)

            features = tf.concat([x, features], 3)

        return features

    '''
    Function for calculate cost volumn is borrowed from 
            - https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/core_costvol.py
            Copyright (c) 2018 Phil Ferriere
            MIT License
            
            which based on 
            - https://github.com/tensorpack/tensorpack/blob/master/examples/OpticalFlow/flownet_models.py
            
            Written by Patrick Wieschollek, Copyright Yuxin Wu
            Apache License 2.0
    '''

    def cost_volumn(self, c1, warp, search_range=4, name='cost_volumn'):
        """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
        Args:
            c1: Level of the feature pyramid of Image1
            warp: Warped level of the feature pyramid of image22
            search_range: Search range (maximum displacement)
        """
        padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
        _, h, w, _ = tf.unstack(tf.shape(c1))
        max_offset = search_range * 2 + 1

        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
                cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
                cost_vol.append(cost)
        cost_vol = tf.concat(cost_vol, axis=3)
        cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

        return cost_vol

    def flow_estimator(self, feature_1, feature_2, up_flow, up_feat, level, num_filters=[128, 128, 96, 64, 32]):
        if up_flow == None and up_feat == None: # the first level for predicting flow 
            corr = self.cost_volumn(feature_1, feature_2)

            x = corr
            x = self.residual_block(x, num_filters, level)
        else:
            warp = bilinear_warp(feature_2, up_flow*self.flow_scales[level])
            corr = self.cost_volumn(feature_1, warp)

            x = tf.concat([corr, feature_1, up_flow, up_feat], 3)
            x = self.residual_block(x, num_filters, level)

        if level > self.output_level: 
            flow = tf.layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='predict_flow_{}'.format(level))(x)

            up_flow = tf.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same', name='up_flow_{}'.format(level))(flow)
            up_feat = tf.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same', name='up_feat_{}'.format(level))(x)

            return flow, up_flow, up_feat
        else: # Final layer
            flow = tf.layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='predict_flow_{}'.format(level))(x)

            return flow, x

    def context_network(self, feature, flow):
        x = tf.layers.Conv2D(filters=128, kernel_size=3, strides=1,  dilation_rate=1, kernel_initializer='he_normal', padding='same', name='dc_conv1')(feature)
        x = tf.nn.leaky_relu(x, 0.1)
        x = tf.layers.Conv2D(filters=128, kernel_size=3, strides=1,  dilation_rate=2, kernel_initializer='he_normal', padding='same', name='dc_conv2')(x)
        x = tf.nn.leaky_relu(x, 0.1)
        x = tf.layers.Conv2D(filters=128, kernel_size=3, strides=1,  dilation_rate=4, kernel_initializer='he_normal', padding='same', name='dc_conv3')(x)
        x = tf.nn.leaky_relu(x, 0.1)
        x = tf.layers.Conv2D(filters=96,  kernel_size=3, strides=1,  dilation_rate=8, kernel_initializer='he_normal', padding='same', name='dc_conv4')(x)
        x = tf.nn.leaky_relu(x, 0.1)
        x = tf.layers.Conv2D(filters=64,  kernel_size=3, strides=1, dilation_rate=16, kernel_initializer='he_normal', padding='same', name='dc_conv5')(x)
        x = tf.nn.leaky_relu(x, 0.1)
        x = tf.layers.Conv2D(filters=32,  kernel_size=3, strides=1,  dilation_rate=1, kernel_initializer='he_normal', padding='same', name='dc_conv6')(x)
        x = tf.nn.leaky_relu(x, 0.1)
        x = tf.layers.Conv2D(filters=2,   kernel_size=3, strides=1,  dilation_rate=1, kernel_initializer='he_normal', padding='same', name='dc_conv_flow')(x)
   
        return x + flow

    def build_graph(self, inputs):
        im1 = inputs[:, :, :, :3]
        im2 = inputs[:, :, :, 3:]

        # Feature pyramids: from deep (6th) to shallow (2nd) features.
        with tf.variable_scope('feature_extractor'):
            ft_pyramids_1 = self.feature_extractor(im1) 

        with tf.variable_scope('feature_extractor', reuse=True):
            ft_pyramids_2 = self.feature_extractor(im2)

        # Flow estimators 
        with tf.variable_scope('flow_estimator'):
            # levels are in descending order -> [n_levels, n_levels-1, n_levels-2, ..., output_level]
            #       level 6   <------->   level 2
            #        deep                 shallow
            # (h/64, w/64, 2)           (h/4, w/4, 2)
            levels = range(self.n_levels, self.output_level-1, -1)

            up_flow, up_feat = None, None 
            flow_pyramids = []
            for i, l in enumerate(levels):
                if l > self.output_level:
                    flow, up_flow, up_feat = self.flow_estimator(feature_1=ft_pyramids_1[i],
                                                                feature_2=ft_pyramids_2[i],
                                                                up_flow=up_flow,
                                                                up_feat=up_feat,
                                                                level=l)

                    flow_pyramids.append(flow)
                else:
                    flow, feat = self.flow_estimator(feature_1=ft_pyramids_1[i],
                                                                feature_2=ft_pyramids_2[i],
                                                                up_flow=up_flow,
                                                                up_feat=up_feat,
                                                                level=l)
            with tf.variable_scope('context_network'):
                final_flow = self.context_network(feature=feat, flow=flow)

            flow_pyramids.append(final_flow)
            # flow_pyramids = [flow6, flow5, flow4, flow3, flow2]

            return flow_pyramids

            # For illustration! 
            # flow6, up_flow6, up_feat6 = self.flow_estimator(feature_1=ft_pyramids_1[0], 
            #                                                 feature_2=ft_pyramids_2[0],
            #                                                 level=6)

            # flow5, up_flow5, up_feat5 = self.flow_estimator(feature_1=ft_pyramids_1[1], 
            #                                                 feature_2=ft_pyramids_2[1],
            #                                                 level=5, 
            #                                                 up_flow=up_flow6,
            #                                                 up_feat=up_feat6)

            # flow4, up_flow4, up_feat4 = self.flow_estimator(feature_1=ft_pyramids_1[2], 
            #                                                 feature_2=ft_pyramids_2[2],
            #                                                 level=4, 
            #                                                 up_flow=up_flow5,
            #                                                 up_feat=up_feat5)

            # flow3, up_flow3, up_feat3 = self.flow_estimator(feature_1=ft_pyramids_1[3], 
            #                                                 feature_2=ft_pyramids_2[3],
            #                                                 level=3, 
            #                                                 up_flow=up_flow4,
            #                                                 up_feat=up_feat4)

            # flow2, _,  feat2 = self.flow_estimator(feature_1=ft_pyramids_1[4], 
            #                                     feature_2=ft_pyramids_2[4],
            #                                     level=2, 
            #                                     up_flow=up_flow3,
            #                                     up_feat=up_feat3)


        
        