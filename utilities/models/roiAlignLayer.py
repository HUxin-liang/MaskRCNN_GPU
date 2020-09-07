import numpy as np
import torch
from torch.autograd import Variable
from utilities.models.utils import log2

############################################################
#  ROIAlign Layer
############################################################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [channels,height, width]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]
    Output:
    Pooled regions in the shape: [num_boxes, channels, height, width,].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # currently only support batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0) # 去掉batchsize那个维度

    # crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI area
    y1, x1, y2, x2 = boxes.chunk(4, dim=1) # 将boxes沿着类方向分成四个块
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    # 224x224大小的ROI对应P4
    # 112x112大小的ROI对应：4+log2( sqrt(112*112)) / (224/sqrt(224*224) ) = 4+-1=3
    # 即 112x112的ROI对应P3
    image_area = Variable(torch.FloatTensor([float(image_shape[0] * image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    roi_level = roi_level.round().int() # round()向下取整
    roi_level = roi_level.clamp(2, 5) # #FPN产生了[P2,P3,P4,P5,P6]五个特征层，但是只有[P2,P3,P4,P5]进行了roi_pooling
    # roi_level: [num_boxes, 1(roi_level)]


    # Loop through levels and apply ROI pooling to each. P2 to P5
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        # i = 0, 1, 2, 3
        # level = 2, 3, 4, 5
        ix = roi_level == level # bool
        if not ix.any(): # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True
            continue
        ix = torch.nonzero(ix)[:, 0] # torch.nonzero() 输出的非零元素在矩阵中的下标矩阵 维度(m,n1,n2,...)
                                     # m是非零元素的个数，n1,n2,...则是ix除了第一维度的其他维度(对于一维的行向量,则为1)
                                     # 假设 ix=tensor([0,1,0,1])  >>> 输出tensor([[1],[3]])  维度(2,1)
                                     # 假设 ix=tensor([[0,1,0,1],[0,1,0,1]]) >>>输出tensor([[0,1],[0,3],[1,0],[1,3]]) 维度(4,2)
        # 因为ix的维度是[n,1],torch.nonzero(ix)生成的维度是(m,2),而通过[:,0],生成的则是,ix中不为0的index
        level_boxes = boxes[ix.data, :] # 分别取出roi_level为2,3,4,5的roi

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data) # 将roi的roi_level为2,3,4,5的index依次append到box_to_level中

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach() # 将抽离导数计算图，也就是说在进行反向传播是不进行求导

        # crop and resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  # CropAndResizeFunction needs batch dimension
        # TODO 完犊子了
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)


