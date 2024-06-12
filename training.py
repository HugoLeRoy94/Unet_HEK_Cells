import tensorflow as tf
from tensorflow.keras import layers, models

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def unet_model(input_size=(256, 256, 1)):
    inputs = tf.keras.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    
    
    return model

# Create the model
model = unet_model()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt

#Folder_SNCA = "/home/hcleroy/PostDoc/Colab_David/ExperimentalData/Processed_Cell_Crops/a-Synuclein_Channel/"
docker_folder_snca = "/app/data_snca/"
SNCAfiles = ['Cell'+str(i)+'_ROI'+str(i)+'_SNCA.tif' for i in range(1,11)]
#Folder_SNCAIP = "/home/hcleroy/PostDoc/Colab_David/ExperimentalData/Processed_Cell_Crops/Synphillin_Channel/"
docker_folder_sncaip = "/app/data_sncaip/"
SNCAIPfiles = ['Cell'+str(i)+'_ROI'+str(i)+'_SNCAIP.tif' for i in range(1,11)]
#Folder_path = "/home/hcleroy/PostDoc/Colab_David/ExperimentalData/Processed_Cell_Crops/mask/"
docker_folder_mask = "/app/data_mask/"
cytoplasms_mask = ['cyt'+str(i)+'.tiff' for i in range(10)]  
condensates_mask = ['cond'+str(i)+'.tiff' for i in range(10)]
cell_mask = ['cell'+str(i)+'.tiff' for i in range(10)]
three_colored = ['three_labeled'+str(i)+'.tif' for i in range(10)]

resolution = 256

# Import and preprocess the SNCA images
snca = [tif.imread(docker_folder_snca + file) for file in SNCAfiles]
snca = [tf.image.resize(stack[:, :, :, np.newaxis], (resolution,resolution)) for stack in snca]
snca = np.array([np.log(img) for stack in snca for img in stack], dtype=np.float32) # add a logscale
snca /= 255.0  # Normalize pixel values to [0, 1]

#sncaip = [tif.imread(docker_folder_sncaip + file) for file in SNCAIPfiles]
#sncaip = [tf.image.resize(stack[:, :, :, np.newaxis], (256,256)) for stack in sncaip]
#sncaip = np.array([img for stack in sncaip for img in stack], dtype=np.float32)
#sncaip /= 255.0  # Normalize pixel values to [0, 1]
# Import and preprocess the cell masks
cells = [tif.imread(docker_folder_mask + file) for file in condensates_mask]
#cells = [tf.image.resize(stack[:, :, :, :], (resolution,resolution)) for stack in cells]
cells = [tf.image.resize(stack[:, :, :, np.newaxis], (resolution,resolution)) for stack in cells]
cells = np.array([img for stack in cells for img in stack], dtype=bool)
#cells /= 255.0  # Normalize pixel values to [0, 1]


# Ensure the shapes are correct
print("Shape of snca:", snca.shape)
#print("Shape of sncaip:", sncaip.shape)
print("Shape of cells:", cells.shape)

# Combine snca and sncaip
combined_images = snca #np.concatenate((snca, sncaip), axis=0)
# Repeat masks to match the combined images
combined_masks = cells #np.concatenate((cells, cells), axis=0)

# Split data into training and validation sets
split_idx = int(0.8 * len(combined_images))
train_images, val_images = combined_images[:split_idx], combined_images[split_idx:]
train_masks, val_masks = combined_masks[:split_idx], combined_masks[split_idx:]


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
batch_size = 4
# Create data generators for training
train_image_generator = image_datagen.flow(train_images, batch_size=batch_size, seed=seed)
train_mask_generator = mask_datagen.flow(train_masks, batch_size=batch_size, seed=seed)
train_generator = zip(train_image_generator, train_mask_generator)

# Create data generators for validation
val_image_generator = image_datagen.flow(val_images, batch_size=batch_size, seed=seed)
val_mask_generator = mask_datagen.flow(val_masks, batch_size=batch_size, seed=seed)
val_generator = zip(val_image_generator, val_mask_generator)
# Fit the model
history = model.fit(train_generator, steps_per_epoch=len(train_images) // 5, epochs=10, validation_data=val_generator, validation_steps=len(val_images) // 5)
model.save('/app/UNET_condensate.h5')