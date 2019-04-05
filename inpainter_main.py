from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from inpainter_utils.pconv2d_data import DataGenerator, torch_preprocessing
from inpainter_utils.pconv2d_model import pconv_model

# SETTINGS:
IMG_DIR_TRAIN   = "data/images/train/"
IMG_DIR_VAL     = "data/images/validation/"
VGG16_WEIGHTS   = 'data/vgg16_weights/vgg16_pytorch2keras.h5'
WEIGHTS_DIR     = "callbacks/weights/"
TB_DIR          = "callbacks/tensorboard/"
CSV_DIR         = 'callbacks/csvlogger/'
BATCH_SIZE      = 4
STEPS_PER_EPOCH = 5000
EPOCHS_STAGE1   = 80
EPOCHS_STAGE2   = 20
LR_STAGE1       = 0.0002
LR_STAGE2       = 0.00005
STEPS_VAL       = 100
BATCH_SIZE_VAL  = 4
IMAGE_SIZE      = (512, 512)
STAGE_1         = True # Initial training if True, Fine-tuning if False 
LAST_CHECKPOINT = "callbacks/weights/initial/weights.80-1.86-1.73.hdf5" # set this to be the path to the checkpoint from the last 
                                                                        # epoch on Stage 1, only needed if STAGE_1 was set to False 

# DATA GENERATORS:
train_datagen   = DataGenerator(preprocessing_function=torch_preprocessing, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    IMG_DIR_TRAIN,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
val_datagen   = DataGenerator(preprocessing_function=torch_preprocessing)
val_generator = val_datagen.flow_from_directory(
    IMG_DIR_VAL,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_VAL,
    seed=22,
    mask_init_seed=1,
    total_steps=STEPS_VAL,
    shuffle=False
)

# TRAINING:
if STAGE_1:
    # Stage 1: initial training
    model = pconv_model(lr=LR_STAGE1, image_size=IMAGE_SIZE, vgg16_weights=VGG16_WEIGHTS)
    model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS_STAGE1,
        validation_data=val_generator,
        validation_steps=STEPS_VAL,
        callbacks=[
            CSVLogger(CSV_DIR + 'initial/log.csv', append=True),
            TensorBoard(log_dir=TB_DIR + 'initial/', write_graph=True),
            ModelCheckpoint(WEIGHTS_DIR + 'initial/weights.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_weights_only=True)
        ]
    )
else:
    # Stage 2: fine-tuning
    model = pconv_model(fine_tuning=True, lr=LR_STAGE2, image_size=IMAGE_SIZE, vgg16_weights=VGG16_WEIGHTS)
    model.load_weights(LAST_CHECKPOINT)
    model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        initial_epoch=EPOCHS_STAGE1,
        epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2,
        validation_data=val_generator,
        validation_steps=STEPS_VAL,
        callbacks=[
            CSVLogger(CSV_DIR + 'fine_tuning/log.csv', append=True),
            TensorBoard(log_dir=TB_DIR + 'fine_tuning/', write_graph=True),
            ModelCheckpoint(WEIGHTS_DIR + 'fine_tuning/weights.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_weights_only=True)
    ]
)
