# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:38:51 2020

@author: Public_1080
"""

import efficientnet.keras as efn
import PIL.Image as Image
import numpy as np
from keras import layers, Model

def test_resize(img_route, input_shape, proc_img = True):
    image = Image.open(img_route) #RGB but show onli size
    iw, ih = image.size
    h, w = input_shape
    
    # resize image 做縮放，可以以其他方法改變
    scale = min(w/iw, h/ih) #選取最小邊
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data=0 #開一個暫存
    if proc_img:
        image = image.resize((nw,nh), Image.BICUBIC) #双立方滤波。在输入图像的4*4矩阵上进行立方插值
        new_image = Image.new('RGB', (w,h), (128,128,128))#開啟一個新圖像，放置128數值
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)/255.
        #image_data = image_data[np.newaxis,:,:,:]
        image_data = np.expand_dims(image_data,axis=0)
    return (image_data)#反回資料並以灰階表示

input_shape = (224,224)

model = efn.EfficientNetB7(weights='imagenet',  include_top=False)
x = model.output
x = layers.GlobalAveragePooling2D()(x)

model = Model(inputs=model.input, outputs=x)

import os 


img_path = 'train/train/17/'
files = os.listdir(img_path)

imgs = []
features = []
for f in files:
    img = test_resize(img_path + f, input_shape)
    imgs.append(img_path + f)
    features.append(model.predict(img))
features = np.array(features)
features = np.reshape(1553, 2560)
from  sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

import pandas as pd
import shutil
data = pd.DataFrame({"files": files, "class" : kmeans.labels_})
imgpath = 'train/train/17/'
cluster_path = 'train/cluster/17/'
for i in range(len(data)):
    img = data.iloc[i][0]
    label = data.iloc[i][1]
    shutil.copy(imgpath + img, cluster_path + str(label) + '/' +  img)