import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder (Downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder (Upsampling)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(4, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

model = unet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

SNCAIPfiles = ['/sncaip/Cell'+str(i)+'_ROI'+str(i)+'_SNCAIP.tif' for i in range(1,11)]
masks = [np.load('/masks/masks4D_'+str(i)+'.npy',allow_pickle=True) for i in range(10)]

resolution = 256

# Import and preprocess the SNCA images
sncaip = [np.load(file) for file in SNCAIPfiles]
sncaip = [tf.image.resize(stack[:, :, :, np.newaxis], (resolution,resolution)) for stack in sncaip]
sncaip = np.array([np.log(img) for stack in sncaip for img in stack], dtype=np.float32) # add a logscale
sncaip /= 255.0  # Normalize pixel values to [0, 1]

# Import and preprocess the cell masks
cells = [np.load(file, allow_pickle=True) for file in masks]
cells = [tf.image.resize(tf.convert_to_tensor(stack[:, :, :, :]), (resolution,resolution)) for stack in cells]
cells = np.array([img for stack in cells for img in stack], dtype=int)

# Ensure the shapes are correct
print("Shape of snca:", sncaip.shape)
print("Shape of cells:", cells.shape)

# Split data into training and validation sets
split_idx = int(0.8 * len(sncaip))
train_images, val_images = sncaip[:split_idx], sncaip[split_idx:]
train_masks, val_masks = masks[:split_idx], masks[split_idx:]

# Define data augmentation for images and masks
image_datagen = ImageDataGenerator(rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
mask_datagen = ImageDataGenerator(rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0.
)

seed = 1
batch_size = 10
# Create data generators for training
train_image_generator = image_datagen.flow(train_images, batch_size=batch_size, seed=seed)
train_mask_generator = mask_datagen.flow(train_masks, batch_size=batch_size, seed=seed)
train_generator = zip(train_image_generator, train_mask_generator)

# Create data generators for validation
val_image_generator = image_datagen.flow(val_images, batch_size=batch_size, seed=seed)
val_mask_generator = mask_datagen.flow(val_masks, batch_size=batch_size, seed=seed)
val_generator = zip(val_image_generator, val_mask_generator)

# Fit the model
history = model.fit(train_generator, steps_per_epoch=len(train_images) // batch_size, epochs=10, validation_data=val_generator, validation_steps=len(val_images) // batch_size)

model.save('/UNET_3Cat.h5')