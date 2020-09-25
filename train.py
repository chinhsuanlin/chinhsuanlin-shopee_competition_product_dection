# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:36:45 2020

@author: Public_2080

"""


"""
Utils
"""
from PIL import Image

"""
Core
"""

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import efficientnet.keras as efn 
from keras.layers import Dense , Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import adam, SGD
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

"""
Callbacks
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
log_dir = 'logs/B4/omega/grab/'
logging = TensorBoard(log_dir=log_dir)
#ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-train_loss{train_loss:.3f}.h5',
#    monitor='train_loss', save_weights_only=True, save_best_only=True, period=3)
checkpoint_acc = ModelCheckpoint(log_dir + 'best_acc{epoch:03d}.h5',
    monitor='val_acc', save_weights_only=False, save_best_only=True, period=1)

checkpoint_loss = ModelCheckpoint(log_dir + 'min_loss.h5',
    monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1)


print('Starting')
dropout_rate = 0.2
input_shape = (224, 224)
n_classes = 42
#base_model = efn.EfficientNetB4(weights='imagenet',  include_top=False)
#model = load_model('path/to/model.h5')
#model = efn.EfficientNetB4(weights=None,  include_top=True, classes=n_classes)





"""
Modify Model to Custom data
"""
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dropout(dropout_rate, name='top_dropout')(x)
#predictions = Dense(n_classes, activation='softmax', name='probs')(x)
#
#model = Model(inputs=base_model.input, outputs=predictions)
#for layer in base_model.layers:
#    layer.trainable = False
    



"""
Data Prepocessing
"""

annotation_path = 'crab_img_df_ver1.txt'
with open (annotation_path, 'r') as f:
    lines = f.readlines()
#lines = lines[:99]
import random
random.shuffle(lines)
num_train = len(lines)
num_val = num_train


def data_generator(annotation_lines, batch_size, input_shape):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        label_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, label = get_random_data(annotation_lines[i], input_shape)
            image_data.append(image)
            label_data.append(label)
            i = (i+1) % n
        image_data = np.array(image_data)
        from keras.utils import to_categorical
        label_data = to_categorical(label_data, num_classes=42)
#        yield image_data, label_data, np.zeros(batch_size)
        yield image_data, label_data
#        yield ( [image_data, *label_data], np.zeros(batch_size) )

def data_generator_wrapper(annotation_lines, batch_size, input_shape):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape)

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a 

def get_random_data(annotation_lines, input_shape, proc_img = True, random =True,
                    flip = False, hue = .1, jitter = .3, sat = .25, val = .50):
    line = annotation_lines.split()
    image = Image.open(line[0])
    label = line[1]
    iw, ih = image.size
    h, w = input_shape
    
    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.
            
        return (image_data, label)
    else:
        """
        Data-Augmentation
        """
    # =============================================================================
    #                               Scale(parameter is 'jitter')
    # =============================================================================
        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
    
    # =============================================================================
    #                               Shift
    # =============================================================================
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
    # =============================================================================
    #                               Flip
    # =============================================================================
        # flip image or not
    #    flip = True
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # =============================================================================
    #                               Color Augmentation
    #                       RGB --> HSV(parameter is hue) --> RGB
    # =============================================================================
        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x) # numpy array, 0 to 1
    
        return (image_data, label)

#aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
#	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#	horizontal_flip=True, fill_mode="nearest")

def plot(history):
    # =============================================================================
    #                   Plot Training Information
    # =============================================================================
    import matplotlib.pyplot as plt
    #define filepath parms
    name = ['freezy', 'unfreezy']
    for index, info in enumerate(history):
        loss_trend_graph_path = log_dir + "loss_" + name[index] + ".jpg" 
        print("Now,we start drawing the loss trends graph...")
        #summarize history for accuracy 
        fig = plt.figure(1)
        plt.plot(info.history["loss"])  
        #plt.plot(history.history["val_acc"])  
        plt.title("Model Loss_" + name[index])  
        plt.ylabel("loss")  
        plt.xlabel("epoch")  
        plt.legend(["train"],loc="upper left")  
        plt.savefig(loss_trend_graph_path) 
        plt.close(1)
        
from keras.models import load_model
model = load_model('grab_021_val9147.h5')

# Frezz
for layer in model.layers[:467]:
   layer.trainable = False
"""
Two-Stage training trick
#"""
# Stage 1 
#model.compile(optimizer=adam(lr=1e-3), loss='categorical_crossentropy', metrics = ['accuracy'])
#batch_size = 256
#print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
#history_freeze = model.fit_generator(data_generator_wrapper(lines, batch_size, input_shape),
#        steps_per_epoch=max(1, num_train//batch_size),
#        validation_data=data_generator_wrapper(lines, batch_size, input_shape),
#        validation_steps=max(1, num_val//batch_size),
#        epochs=2,
#        initial_epoch=0,
#        callbacks=[logging, checkpoint_acc, reduce_lr, early_stopping])
#model.save(log_dir + 'trained_model_stage_1.h5')

#np.argmax(model.predict(imgs), axis=1)


# Stage 2
batch_size = 20

# Unfrezz 
for layer in model.layers[:]:
   layer.trainable = True

model.compile(optimizer=SGD(lr=1e-3, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
history_unfreeze = model.fit_generator(data_generator_wrapper(lines, batch_size, input_shape),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines, batch_size, input_shape),
        validation_steps=max(1, num_train//batch_size),
        epochs=200,
        initial_epoch=22,
        callbacks=[logging, checkpoint_acc, reduce_lr, early_stopping])
model.save(log_dir + 'trained_model_stage_2.h5')

