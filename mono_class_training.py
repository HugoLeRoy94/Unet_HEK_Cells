import tensorflow as tf
from tensorflow.keras import layers, models

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#def unet_model(input_size=(256, 256, 1)):
#    inputs = tf.keras.Input(input_size)
#
#    # Encoder
#    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
#    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
#    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
#
#    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
#    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
#    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
#
#    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
#    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
#    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
#
#    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
#    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
#    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
#
#    # Bottleneck
#    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
#    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
#
#    # Decoder
#    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv5))
#    merge6 = layers.concatenate([conv4, up6], axis=3)
#    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
#    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
#
#    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
#    merge7 = layers.concatenate([conv3, up7], axis=3)
#    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
#    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
#
#    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
#    merge8 = layers.concatenate([conv2, up8], axis=3)
#    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
#    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
#
#    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
#    merge9 = layers.concatenate([conv1, up9], axis=3)
#    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
#    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
#    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same')(conv9)
#
#    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
#
#    model = models.Model(inputs=[inputs], outputs=[outputs])
#
#    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    
#    
#    return model

# https://youtu.be/jvZm8REF2KY
"""
Standard Unet
Model not compiled here, instead will be done externally to make it
easy to test various loss functions and optimizers. 
"""


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K




################################################################
def multi_unet_model(n_classes=1, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    #outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model
 

# Create the model
model = multi_unet_model()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

SNCAIPfiles = ['sncaip/sncaip_'+str(i)+'.npy' for i in range(10)]
masks = ['masks/mask_mono_'+str(i)+'.npy' for i in range(10)]


resolution = 256


# Import and preprocess the SNCA images
sncaip = [np.load(file,allow_pickle=True) for file in SNCAIPfiles]
sncaip = [tf.image.resize(stack[:, :, :, np.newaxis], (resolution,resolution)) for stack in sncaip] # Normalize pixel values to [0, 1] and adjust resolution
sncaip = np.array([img for stack in sncaip for img in stack], dtype=np.float32) # add a logscale
sncaip/=255

# Import and preprocess the cell masks
masks = [np.load(file, allow_pickle=True) for file in masks]
masks = [tf.image.resize(tf.convert_to_tensor(stack[:, :, :,np.newaxis])  , (resolution,resolution)) for stack in masks]
masks = np.array([img for stack in masks for img in stack], dtype=bool)



# Ensure the shapes are correct
print("Shape of snca:", sncaip.shape)
#print("Shape of sncaip:", sncaip.shape)
print("Shape of cells:", masks.shape)


## Split data into training and validation sets
#split_idx = int(0.8 * len(sncaip))
#train_images, val_images = sncaip[:split_idx], sncaip[split_idx:]
#train_masks, val_masks = masks[:split_idx], masks[split_idx:]
#
#
## Define data augmentation for images and masks
#image_datagen = ImageDataGenerator(rotation_range=90,
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    vertical_flip=True,
#    fill_mode='nearest'
#)
#mask_datagen = ImageDataGenerator(rotation_range=90,
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    vertical_flip=True,
#    fill_mode='constant',
#    cval=0.
#)
#
#seed = 1
#batch_size = 5
#
## Custom generator to yield (image, mask) pairs
#def custom_generator(image_generator, mask_generator):
#    while True:
#        image_batch = next(image_generator)
#        mask_batch = next(mask_generator)
#        yield (image_batch, mask_batch)
#
## Create data generators for training
#train_image_generator = image_datagen.flow(train_images, batch_size=batch_size, seed=seed)
#train_mask_generator = mask_datagen.flow(train_masks, batch_size=batch_size, seed=seed)
#train_generator = custom_generator(train_image_generator, train_mask_generator)
#
## Create data generators for validation
#val_image_generator = image_datagen.flow(val_images, batch_size=batch_size, seed=seed)
#val_mask_generator = mask_datagen.flow(val_masks, batch_size=batch_size, seed=seed)
#val_generator = custom_generator(val_image_generator, val_mask_generator)
# Fit the model
callbacks =[
            tf.keras.callbacks.EarlyStopping(patience=3,monitor='val_loss')
]

#history = model.fit(train_generator, steps_per_epoch=len(train_images) // batch_size, epochs=10, validation_data=val_generator, validation_steps=len(val_images) // batch_size)
history = model.fit(sncaip,masks,validation_split=0.1,batch_size=16,epochs=100,callbacks=callbacks)
model.save('UNET_mono_b.h5')