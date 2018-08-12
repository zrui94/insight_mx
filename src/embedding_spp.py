import argparse
import cv2
import numpy as np
import sys
import mxnet as mx
import datetime
from skimage import transform as trans
import sklearn
from sklearn import preprocessing
import datetime
import mxnet as mx

class Embedding:
  def __init__(self, prefix, epoch, layer_name, bt_size, ctx_id=0):
    print('loading',prefix, epoch)
    ctx = mx.gpu(ctx_id)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    #print all_layers
    sym = all_layers[layer_name+'_output'] # fc1_output
    image_size = (112,112)
    self.image_size = image_size
    self.batch_size = bt_size
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model = mx.mod.Module(symbol=sym, context=ctx)
    model.bind(for_training=False, data_shapes=[('data', (self.batch_size, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    
  def get_output(self, batch_data):
      db = mx.io.DataBatch(data=(batch_data,))
      #time_s1 = datetime.datetime.now()
      self.model.forward(db, is_train=False)    
      embedding = self.model.get_outputs()[0]
      embbdding_nd = mx.nd.empty((self.batch_size, 82190))
      for i in range(self.batch_size):
          # print(embedding[i])
          embbdding_nd[i][:] = embedding[i]
          # embbdding_nd[i][:] = embedding[i].flatten()
          # print type(embedding[i])
      return embedding
  
  def get(self, img_name_list):
    input_blob = mx.nd.empty((self.batch_size, 3, self.image_size[1], self.image_size[0]))
    for img_id, img_name in enumerate(img_name_list):
        img_cv = cv2.imread(img_name)
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = mx.nd.array(img)
        img_crop, rect  = mx.image.random_crop(img, (112, 112))
        img_pro = mx.nd.transpose(img_crop, (2,0,1)) #3*112*112, RGB
        #print '-----------------', img_pro.shape
        input_blob[img_id][:] = img_pro
      
    db = mx.io.DataBatch(data=(input_blob,))
    
    for i in range(1):
        time_s1 = datetime.datetime.now()
        self.model.forward(db, is_train=False)    
        embedding = self.model.get_outputs()[0].asnumpy()
        print 'forward time:', datetime.datetime.now() - time_s1
        print embedding.shape
        
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding
    
if __name__ == '__main__':
    model = 'm1'
    if model == 'y1':
        model_prefix = './y1/model-y1-arcface'
        model_epoch = 144
    else:
        model_prefix = './m1/model-m1-arcface-scratch'
        model_epoch = 125
        
    gpu_id = 3
    layer = 'fc1n' #'spp_fc7'
    batch_size = 1 
    
    face_img = './test_img/pol000007_0.725443_pol000007_0005_0012_0_.jpg'
    face_name_list = []
    for i in range(batch_size):
        face_name_list.append(face_img)
    
    em = Embedding(model_prefix, model_epoch, layer, batch_size, gpu_id)
    ft = em.get(face_name_list)
    
    #print ft
    print type(ft)
    print len(ft)
    




