# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:03:20 2020

@author: user
"""
import numpy as np
import efficientnet.keras as efn
from keras.models import load_model
import os
import PIL.Image as Image

input_shape = (224, 224)
n_classes = 42

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


path = 'test/test/'
model = load_model('grab_033_val9294.h5')
files = os.listdir(path)

import pandas as pd
"""
Read Csv
"""
data = pd.read_csv('test.csv')
LABEL = []
NAME = []
count = 0
GROUND = []


#for i in range(42):
#    if i <10:
#        path = 'train/train/0' + str(i) + '/'
#        files = os.listdir(path)
#    else:
#        path = 'train/train/' + str(i) + '/'
#        files = os.listdir(path)
for f in data["filename"]:
#    NAME.append(f)
    img_route=(path + f)
    img = test_resize(img_route, input_shape)
    label = np.argmax(model.predict(img),-1)
    label = label[0]
    if label < 10:
        label = '0' + str(label)
    else:
        label = str(label)
    LABEL.append(label)
#    GROUND.append(i)
    if (count % 1000) == 0:
        print(count)
    count +=1
         
        

data = pd.DataFrame({'filename':data["filename"],'LABEL':LABEL})
data.to_csv('predicts.csv',index=False)

#import pandas as pd
predict_test = pd.read_csv('predicts.csv')
predict_test = predict_test.rename(columns={"filename": "filename", "LABEL": "category"})
predict_test["category"] = predict_test["category"].apply(lambda x: "{:02}".format(x))
predict_test.to_csv("grab_033_val9294.csv", index=False)

"""
Read Csv
"""
#data = pd.read_csv('predict_train_imagenet.csv')
#count = 0
#
#for i in range(len(data.NAME)):
#    if data.LABEL[i] == data.Ground[i]:
#        count +=1
#
#cm = pd.crosstab(data.Ground, data.LABEL, rownames=['Predicts'], colnames=['Ground'])
#cm.to_csv('CM.csv',index=False)
#import matplotlib.pyplot as plt
#import seaborn as sns
#plt.figure(figsize = (420,420))
#plt.subplots(figsize=(20,15))
#sns.heatmap(cm,annot=True)