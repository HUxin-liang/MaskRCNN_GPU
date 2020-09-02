import torch
import torch.nn as nn
import numpy as np

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    '''
    :param scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    :param ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    :param shape:[height, width] spatial shape of the feature map over which
            to generate anchors.
    :param feature_stride: Stride of the feature map relative to the image in pixels.
    :param anchor_stride:Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    :return:
    '''
    # 根据anchor大小和高宽比设置anchor大小
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    # 根据anchor大小和长宽比计算高和宽
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    # 计算 特征空间 对应 原图的位置
    shifts_y = np.array(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.array(0, shape[1], anchor_stride) * feature_stride
    # 把 特征空间原图位置 和 高 宽 组合起来
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    # Reshape to get a list of (y, x) and a list of (h, w)
    # TODO




