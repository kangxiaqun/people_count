#from easydict import EasyDict as edict
import numpy as np

anchors = np.array([[5, 10], [13, 18], [19, 35]])
classes = 20
num = 9
num_anchors_per_layer = 3
batch_size = 1
scratch = False

ignore_thresh = .5
momentum = 0.9
decay = 0.0005
learning_rate = 0.001
max_batches = 50200
lr_steps = [40000, 45000]
lr_scales = [.1, .1]
max_truth = 100
mask = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
image_resized = 416   # { 320, 352, ... , 608} multiples of 32

#
# image process options
#
angle = 0
saturation = 1.5
exposure = 1.5
hue = .1
jitter = .3
random = 1
