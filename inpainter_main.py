import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from inpainter_utils.pconv2d_data import DataGenerator, torch_vgg_pp
from inpainter_utils.pconv2d_model import pconv_model

# SETTINGS:
IMG_DIR      = "data/images/"
WEIGHTS_DIR  = "callbacks/weights/"
TB_PATH      = "callbacks/tensorboard/"
CSV_PATH     = 'callbacks/csvlogger/'
BATCH_SIZE   = 2
VAL_STEPS    = 200
INIT_STAGE   = True # fine-tuning stage if False 

# DATA GENERATORS:
# Create training generator
train_datagen   = DataGenerator(preprocessing_function=torch_vgg_pp, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    IMG_DIR + 'train/',
    target_size=(512, 512),
    batch_size=BATCH_SIZE
)
# Create validation generator
val_datagen   = DataGenerator(preprocessing_function=torch_vgg_pp)
val_generator = val_datagen.flow_from_directory(
    IMG_DIR + 'validation/',
    target_size=(512, 512),
    batch_size=BATCH_SIZE,
    seed=22,
    mask_init_seed=1,
    total_steps=VAL_STEPS,
    shuffle=False
)

# TRAINING:
if INIT_STAGE:
    # Stage 1: initial training
    model = pconv_model(lr=0.0001)
    model.fit_generator(
        train_generator,
        steps_per_epoch=10000,
        epochs=50,
        validation_data=val_generator,
        validation_steps=VAL_STEPS,
        callbacks=[
            CSVLogger(CSV_PATH + 'initial/log.csv'),
            TensorBoard(log_dir=TB_PATH + 'initial/', write_graph=True),
            ModelCheckpoint(WEIGHTS_DIR + 'initial/weights.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_weights_only=True)
        ]
    )
else:
    # Stage 2: fine-tuning
    model = pconv_model(fine_tuning=True, lr=0.00005)
    # EZ: Replace the file name!
    model.load_weights(WEIGHTS_DIR + 'fine_tuning/weights.97-2.22.hdf5')
    model.fit_generator(
        train_generator,
        steps_per_epoch=10000,
        initial_epoch=50, # EZ: change to what you've got after Stage 1!
        epochs=50,
        validation_data=val_generator,
        validation_steps=VAL_STEPS,
        callbacks=[
            CSVLogger(CSV_PATH + 'fine_tuning/log.csv'),
            TensorBoard(log_dir=TB_PATH + 'fine_tuning/', write_graph=True),
            ModelCheckpoint(WEIGHTS_DIR + 'fine_tuning/weights.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_weights_only=True)
    ]
)
