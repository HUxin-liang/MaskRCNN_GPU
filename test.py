# class te():
#     def __init__(self, architecture):
#         super(te, self).__init__()
#         assert architecture in ['resnet50', 'resnet101']
#         self.layers = [3, 4, {'resnet50': 6, 'resnet101': 23}[architecture], 3]
#     def print(self):
#         print(self.layers)
#
#
# a = te('resnet101')
# print(a)
# a.print()

# import torch
# a = torch.ones([4,5,6,7])
# print(a.shape)
# b = a[1:]
# print(b.shape)

# for i, level in enumerate(range(2, 6)):
#     print('i:', i)
#     print('level:', level)
import torch

a = torch.Tensor([[[0.6, 0.0, 0.0, 0.0],
                  [0.0, 0.4, 0.0, 0.0],
                  [0.0, 0.0, 1.2, 0.0],
                  [0.0, 0.0, 0.0,-0.4],
                  [0.0, 0.0, 0.0,-0.4]],
                 [[0.6, 0.0, 0.0, 0.0],
                  [0.0, 0.4, 0.0, 0.0],
                  [0.0, 0.0, 1.2, 0.0],
                  [0.0, 0.0, 0.0, -0.4],
                  [0.0, 0.0, 0.0, -0.4]]]
                 )
b = torch.nonzero(a)
print(a.shape)
print(b)
print(b.shape)