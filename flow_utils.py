import os
import re
import numpy as np
import tensorflow as tf

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

'''
Library for manipulate optical flow are borrowed from:
        - https://github.com/liruoteng/OpticalFlowToolkit
        Copyright (c) 2019 LI RUOTENG
        MIT License
'''

def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        # print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def read_pfm_file(flow_file):
    """
    Read from .pfm file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    def readPFM(file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip().decode('utf-8')

        assert header == 'PF'
        
        dim_match = file.readline().decode('utf-8')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', dim_match)

        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip().decode('utf-8'))

        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3)

        data = np.reshape(data, shape)[:, :, :2]
        data = np.flipud(data)
        
        return data, scale

    (data, scale) = readPFM(flow_file)

    return data     

def read_flow(filename):
    """
    read optical flow data from flow file
    :param filename: name of the flow file
    :return: optical flow data in numpy array
    """
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)
    elif filename.endswith('.pfm'):
        flow = read_pfm_file(filename)
    else:
        raise Exception('Invalid flow file format!')

    return flow

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()
    
'''
Codes for warping optical flow are borrowed from:
        - https://github.com/daigo0927/pwcnet
        Copyright (c) 2019 diago0927
        MIT License
'''
def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11