# Image Inpainting Based on Partial Convolutions in Keras
Unofficial implementation of [Liu et al., 2018. Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723).

This implementation was inspired by and is partially based on the early version of [this repository](https://github.com/MathiasGruber/PConv-Keras). Many ideas, e.g. random mask generator using OpenCV, were taken and used here.

## Requirements
- Python 3.6
- TensorFlow 1.13 
- Keras 2.2.4
- OpenCV and NumPy (for mask generator)

## How to run the code
You can either run ```python inpainter_main.py``` to train your model and then test it using the jupyter notebook provided or train and test the model in the jupyter notebook straight away. Make sure you set the paths to your datasets before running the code. Note, the code is using the ImageDataGenerator class from Keras. These paths should therefore point to one level above in the directory tree, i.e. if e.g. your train images are stored in the directory ```path/to/train/images/subdir/``` then you set ```IMG_DIR_TRAIN = path/to/train/images/```. If there are more than one directory in ```path/to/train/images/```, they all will be used in training.

## VGG16 model for feature extraction
The authors of the paper used PyTorch to implement the model. The VGG16 model was chosen for feature extraction. The [VGG16 model in PyTorch](https://pytorch.org/docs/stable/torchvision/models.html) was trained with the following image pre-processing:
1. Divide the image by 255,
2. Subtract [0.485, 0.456, 0.406] from the RGB channels, respectively,
3. Divide the RGB channels by [0.229, 0.224, 0.225], respectively.

The same pre-processing scheme was used in the paper. The [VGG16 model in Keras](https://keras.io/applications/#vgg16) comes with weights ported from the original Caffe implementation and expects another image pre-processing:
1. Convert the images from RGB to BGR,
2. Subtract [103.939, 116.779, 123.68] from the BGR channels, respectively.

Due to different pre-processing, the scales of features extracted using the VGG16 model from PyTorch and Keras are different. If we were to use the build-in VGG16 model in Keras, we would need to modify the loss term normalizations in Eq. 7. To avoid this, the weights of the VGG16 model were ported from PyTorch using [this script](https://github.com/ezavarygin/vgg16_pytorch2keras) and are provided in the file ```data/vgg16_weights/vgg16_pytorch2keras.h5```. The PyTorch-style image pre-processing is used in the code.
<!---
*L<sub>total</sub>* = *L<sub>valid</sub>* + 6*L<sub>hole</sub>* + 0.05*L<sub>perceptual</sub>* + 120(*L<sub>style out</sub>* + *L<sub>style comp</sub>*) + 0.1*L<sub>tv</sub>*
--->

## Mask dataset
Random masks consisting of circles and lines are generated using the OpenCV library. The mask generator is the modified version of the one used [here](https://github.com/MathiasGruber/PConv-Keras). It was modified to generate reproducible masks for validation images.
To generate consistent masks, i.e. the same set of masks after each epochs, for validation images, make sure the number of images in your validation set is equal to the product of ```VAL_STEPS``` and ```VAL_BATCH_SIZE```. I used 400 validation images with ```VAL_STEPS = 100``` and ```VAL_BATCH_SIZE = 4```. You might need to change these parameters if you want to use more/less validation images with consistent masks between different epochs.

## Image dataset
The examples shown below were generated using the model trained on the [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) (subset with bounding boxes). You can train the model using other datasets.

## Training

The model was trained in two steps:
1. Initial training (80 epochs, learning rate 0.0002, BatchNorm enabled),
2. Fine-tuning (20 epochs, learning rate 0.00005, BatchNorm disabled in encoder)

with the batch size of 4 and 5000 steps per epoch.

## Results
Comming soon...

## Comments
In the examples shown above, the trained model performs very well. However, in some cases the results are not even near as good. Likely, more training is needed.
