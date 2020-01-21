import tensorflow as tf
import numpy as np
from scipy import ndimage



def pad_amount(k):
    added = k - 1
    # note: this imitates scipy, which puts more at the beginning
    end = added // 2
    start = added - end
    return [start, end]

def neighborhood(x, kh, kw):
    # input: N, H, W, C
    # output: N, H, W, KH, KW, C
    # padding is REFLECT
    xs = tf.shape(x)
    x_pad = tf.pad(x, ([0, 0], pad_amount(kh), pad_amount(kw), [0, 0]), 'SYMMETRIC')
    return tf.reshape(tf.extract_image_patches(x_pad,
                                               [1, kh, kw, 1],
                                               [1, 1, 1, 1],
                                               [1, 1, 1, 1],
                                               'VALID'),
                      (xs[0], xs[1], xs[2], kh, kw, xs[3]))

def median_filter_tf(x, kh, kw=-1):
    if kw == -1:
        kw = kh
    neigh_size = kh * kw
    xs = tf.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = tf.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = tf.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    x_top, _ = tf.nn.top_k(x_neigh, rank)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    return tf.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))

def median_filter_no_reshape(x, kh, kw):
    neigh_size = kh * kw
    xs = tf.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = tf.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = tf.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    x_top, _ = tf.nn.top_k(x_neigh, rank)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    # return tf.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))
    return x_mid

def median_random_filter_tf(x, kh, kw):
    neigh_size = kh * kw
    xs = tf.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = tf.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = tf.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    rand_int = tf.cast(tf.truncated_normal([1], mean=0, stddev=neigh_size/4)[0], tf.int32)
    x_top, _ = tf.nn.top_k(x_neigh, rank+rand_int)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    return tf.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))

def median_random_filter_no_reshape(x, kh, kw):
    neigh_size = kh * kw
    xs = tf.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = tf.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = tf.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    rand_int = tf.cast(tf.truncated_normal([1], mean=0, stddev=neigh_size/4)[0], tf.int32)
    x_top, _ = tf.nn.top_k(x_neigh, rank+rand_int)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    return x_mid

def median_random_pos_size_filter_tf(x, kh, kw):
    pass
    # Get two/multiple x_mid, randomly select from one .
    s0 = median_random_filter_no_reshape(x, 2, 2)
    s1 = median_random_filter_no_reshape(x, 3, 3)
    s2 = median_random_filter_no_reshape(x, 4, 4)

    xs = tf.shape(x)
    nb_pixels = xs[0] * xs[1] * xs[2] * xs[3]
    samples_mnd = tf.squeeze(tf.multinomial(tf.log([[10., 10., 10.]]), nb_pixels))

    # return tf.constant([0]*nb_pixels, dtype=tf.int64)
    zeros = tf.zeros([nb_pixels], dtype=tf.int64)
    ones = tf.ones([nb_pixels], dtype=tf.int64)
    twos = tf.ones([nb_pixels], dtype=tf.int64)*2
    # tmp = tf.cast(tf.equal(samples_mnd, tf.zeros([nb_pixels], dtype=tf.int64)), tf.int64)
    # return zeros, ones, twos

    selected_0 = tf.cast(tf.equal(samples_mnd, zeros), tf.float32)
    selected_1 = tf.cast(tf.equal(samples_mnd, ones), tf.float32)
    selected_2 = tf.cast(tf.equal(samples_mnd, twos), tf.float32)

    # return s0, selected_0
    x_mid = tf.add_n( [tf.multiply(s0, selected_0), tf.multiply(s1, selected_1), tf.multiply(s2, selected_2)] )

    return tf.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))


def median_random_size_filter_tf(x, kh, kw):
    pass
    # Get two/multiple x_mid, randomly select from one .
    s0 = median_filter_no_reshape(x, 2, 2)
    s1 = median_filter_no_reshape(x, 3, 3)
    # s2 = median_filter_no_reshape(x, 4, 4)

    xs = tf.shape(x)
    nb_pixels = xs[0] * xs[1] * xs[2] * xs[3]
    samples_mnd = tf.squeeze(tf.multinomial(tf.log([[10., 10.]]), nb_pixels))

    # return tf.constant([0]*nb_pixels, dtype=tf.int64)
    zeros = tf.zeros([nb_pixels], dtype=tf.int64)
    ones = tf.ones([nb_pixels], dtype=tf.int64)
    # twos = tf.ones([nb_pixels], dtype=tf.int64)*2
    # tmp = tf.cast(tf.equal(samples_mnd, tf.zeros([nb_pixels], dtype=tf.int64)), tf.int64)
    # return zeros, ones, twos

    selected_0 = tf.cast(tf.equal(samples_mnd, zeros), tf.float64)
    selected_1 = tf.cast(tf.equal(samples_mnd, ones), tf.float64)
    # selected_2 = tf.cast(tf.equal(samples_mnd, twos), tf.float32)

    # return s0, selected_0
    # x_mid = tf.add_n( [tf.multiply(s0, selected_0), tf.multiply(s1, selected_1), tf.multiply(s2, selected_2)] )
    x_mid = tf.add_n( [tf.multiply(s0, selected_0), tf.multiply(s1, selected_1)] )

    return tf.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))




def reduce_precision_py(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    npp_int = npp - 1
    x_int = np.rint(x * npp_int)
    x_float = x_int / npp_int
    return x_float


def reduce_precision_tf(x, npp):
    """
    Reduce the precision of image, the tensorflow version.
    """
    npp_int = npp - 1
    x_int = tf.rint(tf.multiply(x, npp_int))
    x_float = tf.div(x_int, npp_int)
    return x_float


def bit_depth_py(x, bits):
    precisions = 2**bits
    return reduce_precision_py(x, precisions)

def bit_depth_tf(x, bits):
    precisions = 2**bits
    return reduce_precision_tf(x, precisions)


def rotation_py(x, angle):

    #return rotate(x, angle)
    image= ndimage.rotate(x, angle,reshape=False, axes=(1, 2),order=1)

    return image

def rotation_tf(x, angle):

    return tf.contrib.image.rotate(x, angle)

def bit_depth_random_py(x, bits, stddev):
    if stddev == 0.:
        rand_array = np.zeros(x.shape)
    else:
        rand_array = np.random.normal(loc=0., scale=stddev, size=x.shape)
    x_random = np.add(x, rand_array)
    return bit_depth_py(x_random, bits)

def bit_depth_random_tf(x, bits, stddev):
    rand_ts = tf.random_normal(x.get_shape(), mean=0, stddev=stddev)
    x_random = tf.add(x, rand_tf)
    return bit_depth_tf(x_random, bits)

def binary_filter_tf(x, threshold):
    x_bin = tf.nn.relu(tf.sign(x-threshold))
    return x_bin

def binary_filter_py(x, threshold):
    x_bin = np.maximum(np.sign(x-threshold), 0)
    return x_bin

def binary_random_filter_tf(x, threshold, stddev=0.125):
    rand_ts = tf.random_normal(x.get_shape(), mean=0, stddev=stddev)
    x_bin = tf.nn.relu(tf.sign(tf.add(x,rand_ts)-threshold))
    return x_bin

def binary_random_filter_py(x, threshold, stddev=0.125):
    if stddev == 0.:
        rand_array = np.zeros(x.shape)
    else:
        rand_array = np.random.normal(loc=0., scale=stddev, size=x.shape)
    x_bin = np.maximum(np.sign(np.add(x, rand_array)-threshold), 0)
    return x_bin


def median_filter_py(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    return ndimage.filters.median_filter(x, size=(1,width,height,1), mode='reflect')

def median_random_filter_py(x, width, height=-1):
    # assert False
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        x = tf.constant(x)
        res = median_random_filter_tf(x, width, height)
        return res.eval()

def median_random_pos_size_filter_py(x, width, height=-1):
    # assert False
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        x = tf.constant(x)
        res = median_random_pos_size_filter_tf(x, width, height)
        return res.eval()

def median_random_size_filter_py(x, width, height=-1):
    # assert False
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        x = tf.constant(x)
        res = median_random_size_filter_tf(x, width, height)
        return res.eval()


# Inverifiers implemented in OpenCV
# OpenCV expects uint8 as image data type.
def opencv_wrapper(imgs, opencv_func, argv):
    ret_imgs = []
    imgs_copy = imgs

    if imgs.shape[3] == 1:
        imgs_copy = np.squeeze(imgs)

    for img in imgs_copy:
        img_uint8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
        ret_img = opencv_func(*[img_uint8]+argv)
        if type(ret_img) == tuple:
            ret_img = ret_img[1]
        ret_img = ret_img.astype(np.float32) / 255.
        ret_imgs.append(ret_img)
    ret_imgs = np.stack(ret_imgs)

    if imgs.shape[3] == 1:
        ret_imgs = np.expand_dims(ret_imgs, axis=3)

    return ret_imgs


# Binary filters.
def adaptive_binarize_py(x, block_size=5, C=33.8):
    "Works like an edge detector."
    # ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C
    # THRESH_BINARY, THRESH_BINARY_INV
    import cv2
    ret_imgs = opencv_wrapper(x, cv2.adaptiveThreshold, [255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C])
    return ret_imgs

def otsu_binarize_py(x):
    # func = lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # return opencv_binarize(x, func)
    import cv2
    ret_imgs = opencv_wrapper(x, cv2.threshold, [0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU])
    return ret_imgs


# Non-local Means
def non_local_means_color_py(imgs, search_window, block_size, photo_render):
    import cv2
    ret_imgs = opencv_wrapper(imgs, cv2.fastNlMeansDenoisingColored, [None,photo_render,photo_render,block_size,search_window])
    return ret_imgs

def non_local_means_color_tf(imgs, search_window, block_size, photo_render):
    my_func = lambda x: non_local_means_color_py(x, search_window, block_size, photo_render)
    y = tf.py_func(my_func, [imgs], tf.float32, stateful=False)
    return y

def non_local_means_bw_py(imgs, search_window, block_size, photo_render):
    import cv2
    ret_imgs = opencv_wrapper(imgs, cv2.fastNlMeansDenoising, [None,photo_render,block_size,search_window])
    return ret_imgs

def non_local_means_bw_tf(imgs, search_window, block_size, photo_render):
    my_func = lambda x: non_local_means_bw_py(x, search_window, block_size, photo_render)
    y = tf.py_func(my_func, [imgs], tf.float32, stateful=False)
    return y


def bilateral_filter_py(imgs, d, sigmaSpace, sigmaColor):
    """
    :param d: Diameter of each pixel neighborhood that is used during filtering. 
        If it is non-positive, it is computed from sigmaSpace.
    :param sigmaSpace: Filter sigma in the coordinate space. 
        A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). 
        When d>0, it specifies the neighborhood size regardless of sigmaSpace. 
        Otherwise, d is proportional to sigmaSpace.
    :param sigmaColor: Filter sigma in the color space. 
        A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
    """
    import cv2
    return opencv_wrapper(imgs, cv2.bilateralFilter, [d, sigmaColor, sigmaSpace])

def bilateral_filter_tf(imgs, d, sigmaSpace, sigmaColor):
    my_func = lambda x: bilateral_filter_py(x, d, sigmaSpace, sigmaColor)
    y = tf.py_func(my_func, [imgs], tf.float32, stateful=False)
    return y


# Adaptive Bilateral Filter
# https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#adaptivebilateralfilter
# Removed in OpenCV > 3.0.
def adaptive_bilateral_filter_py(imgs, ksize, sigmaSpace, maxSigmaColor=20.0):
    import cv2
    return opencv_wrapper(imgs, cv2.adaptiveBilateralFilter, [(ksize,ksize), sigmaSpace, maxSigmaColor])

def adaptive_bilateral_filter_tf(imgs, ksize, sigmaSpace, maxSigmaColor=20.0):
    my_func = lambda x: adaptive_bilateral_filter_py(x, ksize, sigmaSpace, maxSigmaColor)
    y = tf.py_func(my_func, [imgs], tf.float32, stateful=False)
    return y


none_tf = none_py = lambda x:x


import sys, os
#project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

from externals.MagNet.worker import SimpleReformer
mnist_autoencoder_fpath = os.path.join(project_path, "downloads/MagNet/defensive_models/MNIST_I")
cifar10_autoencoder_fpath = os.path.join(project_path, "downloads/MagNet/defensive_models/CIFAR")

reformer_mnist = SimpleReformer(mnist_autoencoder_fpath)
reformer_cifar10 = SimpleReformer(cifar10_autoencoder_fpath)

def magnet_mnist_py(imgs):
    return reformer_mnist.heal(imgs)

def magnet_cifar10_py(imgs):
    return reformer_cifar10.heal(imgs)

# Construct a name search function.
def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False

def parse_params(params_str):
    params = []

    for param in params_str.split('_'):
        param = param.strip()
        if param.isdigit():
            param = int(param)
        elif isfloat(param):
            param = float(param)
        else:
            continue
        params.append(param)

    return params

def get_inverifier_by_name(name, func_type):
    inverifier_list = ['none',
                     'bit_depth_random',
                     'bit_depth',
                     'binary_filter',
                     'binary_random_filter',
                     'adaptive_binarize',
                     'otsu_binarize',
                     'median_filter',
                     'median_random_filter',
                     'median_random_size_filter',
                     'non_local_means_bw',
                     'non_local_means_color',
                     'adaptive_bilateral_filter',
                     'bilateral_filter',
                     'magnet_mnist',
                     'magnet_cifar10',
                     'rotation'
                    ]

    for inverifier_name in inverifier_list:
        if name.startswith(inverifier_name):
            func_name = "%s_py" % inverifier_name if func_type=='python' else "%s_tf" % inverifier_name
            params_str = name[len(inverifier_name):]

            # Return a list
            args = parse_params(params_str)
            # print ("params_str: %s, args: %s" % (params_str, args))

            return lambda x: globals()[func_name](*([x]+args))

    raise Exception('Unknown inverifier name: %s' % name)

def get_sequential_inverifiers_by_name(inverifiers_name):
    # example_inverifiers_name = "binary_filter_0.5,median_smoothing_2_2"
    verifier_func = None
    for inverifier_name in inverifiers_name.split(','):
        inverifier = get_inverifier_by_name(inverifier_name, 'python')

        if inverifier_func == None:
            inverifier_func = lambda x: inverifier(x)
        else:
            old_func = inverifier_func
            inverifier_func = lambda x: inverifier(old_func(x))
    return squeeze_func
