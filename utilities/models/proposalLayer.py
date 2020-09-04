import numpy as np
import torch
from torch.autograd import Variable
from utilities.models.nms import nms


#  Proposal Layer


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def clip_boxes(boxes, window):
    """
    将 框 clip到原始图片区域
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.
    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Currently only supports batchsize 1
    # inputs是一个list [rpn_probs,rpn_bbox]
    # inputs[0] 是rpn_probs (batch,anchors_num,2)
    # inputs[1] 是rpn_bbox  (batch,anchors_num,4)
    # 去掉batch_size那个维度, 因为每个batch只支持一张图片
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1] #获得deltas,这是每张图片在第一阶段的预测输出
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0]) #在进行nms之前取出多少个anchors
    scores, order = scores.sort(descending=True) #将scores进行倒序排序，返回排序后的结果，以及他们在原序列的的index
    order = order[:pre_nms_limit] #切片前pre_nms_limit个scores
    scores = scores[:pre_nms_limit] #切片前pre_nms_limit个scores
    deltas = deltas[order.data, :] # TODO: Support batch size > 1 ff. #取出scores排名前pre_nms_limit的deltas
    anchors = anchors[order.data, :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    # 根据输出偏差对anchors进行修正
    # 将取出来的deltas进行变换成方框的左上角和右下角坐标(pre_nms_limit,4)
    # (pre_nms_limit,(y1, x1, y2, x2))
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window) # 将boxes限制在图片边界内

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    # torch.cat((boxes, scores.unsqueeze(1)), 1)将scores增加一个维度变成(pre_nms_limit,1),
    # 与boxes (pre_nms_limit,4)在第dims=1的维度上拼接变成维度为(pre_nms_limit,5)
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold) #keep是一个list，保存的是经过nms后剩下来的box的index
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    # 归一化
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0) # 增加一个维度 (proposal_count,4)--> (1,proposal_count,4)

    return normalized_boxes