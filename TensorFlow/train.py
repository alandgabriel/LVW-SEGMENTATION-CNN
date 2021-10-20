


import tensorflow as tf
import glob
import random
import tensorflow.keras.layers as layers
import numpy as np
from skimage.io import imread
import os
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from packaging import version
import datetime
from tensorboard.plugins.hparams import api as hp
import time
import skimage.io as io
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)


def multi_ero(im):
    Filter = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])
    
    im = erosion(im, Filter)
    return im


def unet():
  inputs = tf.keras.Input((112, 112, 1))

  # Entry block
  x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x  # Set aside residual

  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for filters in [64, 128, 256]:
      x = layers.Activation("relu")(x)
      x = layers.SeparableConv2D(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.Activation("relu")(x)
      x = layers.SeparableConv2D(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

      # Project residual
      residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
          previous_block_activation
      )
      x = layers.add([x, residual])  # Add back residual
      previous_block_activation = x  # Set aside next residual

  ### [Second half of the network: upsampling inputs] ###

  for filters in [256, 128, 64, 32]:
      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.UpSampling2D(2)(x)

      # Project residual
      residual = layers.UpSampling2D(2)(previous_block_activation)
      residual = layers.Conv2D(filters, 1, padding="same")(residual)
      x = layers.add([x, residual])  # Add back residual
      previous_block_activation = x  # Set aside next residual

  # Add a per-pixel classification layer
  outputs = layers.Conv2D(1,(1, 1), activation='sigmoid') (x)

  # Define the model
  model = tf.keras.Model(inputs, outputs)
  return model



class data(tf.keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
    

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size , dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.squeeze (io.imread(path, plugin='simpleitk'))
            x[j] = cv2.resize(img, self.img_size, interpolation = cv2.INTER_AREA)
        y = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = np.squeeze (io.imread(path, plugin='simpleitk'))
            img [(img==3) + (img ==1)] = 0
            img [img==2] = 255
            img = multi_ero(img)
            y[j] = cv2.resize(img, self.img_size, interpolation = cv2.INTER_AREA)
        return x, y
    
device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

tf.debugging.set_log_device_placement(True)


#Detecting GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#import examples for training
'''
folder = 'training'
extension = '.mhd'
files = []
for r, d, f in os.walk(folder):
    for file in f:
        if file.endswith(extension):
            files.append(os.path.join(r, file))

images = [s for s in files if ('sequence' not in s) and ('gt' not in s)] 
masks = [s for s in files if ('sequence' not in s) and ('gt' in s)] 
N = len (images)
'''
images = glob.glob('augmentedData/*.png')
masks = [glob.glob('masks/' + os.path.basename(im))[0] for im in images]
N = len(images)

# Spliting data 

ixRand  = list(range(N))
random.shuffle(ixRand)
train_data = [images[e] for e in ixRand[:round(N*.8)]]
train_targets = [masks[e] for e in ixRand[:round(N*.8)]]

val_data = [images[e] for e in ixRand[round(N*.8):]]
val_targets = [masks[e] for e in ixRand[round(N*.8):]]

# torch needs that data comes from an instance with getitem and len methods (Map-style datasets)

training_dataset = data(32,(112,112), train_data, train_targets)

val_dataset = data(32,(112,112), val_data, val_targets)


model = unet()
model.compile(optimizer='adam',
               loss='binary_crossentropy',
              metrics=['accuracy'])


tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)

callbacks = [
   # tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=False),
    tf.keras.callbacks.ModelCheckpoint('segmentationTrain.h5', monitor='val_loss', save_best_only=True, save_freq = 'epoch'),
    tf.keras.callbacks.CSVLogger('segmentationTrain.csv')
]


# Train the model, doing validation at the end of each epoch.
epochs = 500
start = time.time()
history = model.fit(training_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
end = time.time()
elapsed = end-start
#%%

#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('$Model_{Accuracy}$')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('$Model_{Loss}$') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()



#%% Test model

pred = model.predict(val_dataset)
r = np.random.randint(len(val_data))
echo = np.squeeze (io.imread(val_data[r], plugin='simpleitk'))
echo = cv2.resize(echo, (112,112), interpolation = cv2.INTER_AREA)
mask = np.squeeze (io.imread(val_targets[r], plugin='simpleitk'))
mask = cv2.resize(mask, (112,112), interpolation = cv2.INTER_AREA)
mask [(mask==1) + (mask ==3)] = 0
mask [mask==2] = 255
predic = np.squeeze (pred[6])
predic = predic / np.max(predic) # normalize the data to 0 - 1
predic = 255 * predic # Now scale by 255
predic = predic.astype(np.uint8)
mask = multi_ero(mask)
plt.subplot(121)
plt.imshow(predic + echo/2 )
plt.title('Prediction')
plt.subplot(122)
plt.imshow( echo/2 + mask)
plt.title('Ground trouth')
plt.show()
#%%

folder = 'testing'
extension = '.mhd'
files = []
for r, d, f in os.walk(folder):
    for file in f:
        if file.endswith(extension):
            files.append(os.path.join(r, file))

images = [s for s in files if ('sequence' not in s) and ('gt' not in s)] 
img = np.squeeze (io.imread(images[np.random.randint(len(images))], plugin='simpleitk'))
img = cv2.resize(img, (112,112), interpolation = cv2.INTER_AREA)

plt.subplot(121)
plt.imshow(img)
plt.title('Echocardiogram')
pred = model.predict(img.reshape (1,112,112,1))
plt.subplot(122)
plt.imshow (pred[0,:,:,0])
plt.title('LV wall segmentation')
plt.show()
