import argparse
import cv2
import numpy as np
import sys
import mxnet as mx
from image_iter_kd import FaceImageIter

    
if __name__ == '__main__':
    data_shape_input = (3, 112, 112)
    path_imgrec_input = "/data/shipeipei/model_compression/data/mxnet_0611/train.rec"
    train_dataiter = FaceImageIter(batch_size = 1, data_shape = data_shape_input, path_imgrec = path_imgrec_input, shuffle = True,
                                   rand_mirror = 1, mean = None, cutoff = 0)
                                   
    data_batch_first = train_dataiter.next()
    print data_batch_first

    




