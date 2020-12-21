# coding: utf-8

from __future__ import division, print_function

import cv2
import random
import numpy as np

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    # idx = np.random.randint(0, len(_COLORS))
    idx = 9
    ret = _COLORS[idx] * maximum

    if not rgb:
        ret = ret[::-1]
    ret = ret.astype(np.int32)

    return ret


def gen_random_color(number=80):
    color_table = np.random.randint(0, 255, size=(number, 3))
    return color_table


def plot_one_img(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''

    if color is None:
        color = [random.randint(0, 255) for _ in range(3)]
    else:
        color = [int(color[0]), int(color[1]), int(color[2])]

    if line_thickness is None:
        tl = int(round(0.002 * max(img.shape[0:2])))
    else:
        tl = line_thickness
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))

    cv2.rectangle(img, c1, c2, color, thickness=tl)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


def plot_multi_images(imgs, ret,line_thickness=None):
    for i, element in enumerate(ret):
        for j, dict_ in enumerate(element):
            # x1 = dict_['box'][0] * imgs[i].shape[0]
            # x2 = dict_['box'][2] * imgs[i].shape[0]
            # y1 = dict_['box'][1] * imgs[i].shape[1]
            # y2 = dict_['box'][3] * imgs[i].shape[1]

            x1 = dict_['box'][0] * imgs[i].shape[1]
            x2 = dict_['box'][2] * imgs[i].shape[1]
            y1 = dict_['box'][1] * imgs[i].shape[0]
            y2 = dict_['box'][3] * imgs[i].shape[0]
            # print([x1, y1, x2, y2])

            plot_one_img(imgs[i], [x1, y1, x2, y2],
                         # label=dict_['class_name'] + ', {:.2f}%'.format(dict_['score'] * 100),
                         label='',
                         color=random_color(),
                         line_thickness=line_thickness)


def plot_multi_images_v1(imgs, ret, number=80):
    color_table = gen_random_color(number=number)
    for i, element in enumerate(ret):
        for j, dict_ in enumerate(element):
            x1 = dict_['box'][0] * imgs[i].shape[1]
            x2 = dict_['box'][2] * imgs[i].shape[1]
            y1 = dict_['box'][1] * imgs[i].shape[0]
            y2 = dict_['box'][3] * imgs[i].shape[0]
            color = color_table[dict_['class_id']]
            plot_one_img(imgs[i],
                         [x1, y1, x2, y2],
                         label=dict_['class_name'] + ', {:.2f}%'.format(dict_['score'] * 100),
                         color=color)
