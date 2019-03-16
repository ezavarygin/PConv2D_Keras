from keras.preprocessing.image import ImageDataGenerator
import cv2
from random import randint, seed
import numpy as np


class DataGenerator(ImageDataGenerator):

    def flow_from_directory(self, directory, mask_init_seed=None, total_steps=None, **kwargs):
        """
        Modified flow_from_directory method.
        
        Parameters
            mask_init_seed: int, initial seed to use for random masks.
            total_steps: int, the total number of steps with different masks. Masks 
                from the total_steps+n step are equal to those from the nth step. 
                Should be int > 0 or None.
        Comments
            Full reproducibility of the validation set is achieved by setting `seed` 
                and `mask_init_seed` to some int values, `total_steps` to int value 
                equal to `validation_steps` in model.fit_generator() and `shuffle` 
                to False. `total_steps` and `validation_steps` should also iterate 
                over the entire validation dataset.
        """
        generator = super().flow_from_directory(directory, class_mode=None, **kwargs)
        
        if mask_init_seed is not None:
            assert isinstance(mask_init_seed, int), "`mask_init_seed` must be None or int."
            assert isinstance(total_steps, int) and total_steps > 0, "if `mask_init_seed` is int, `total_steps` must be int > 0."
            seed_add = 0
            
        while True:
            
            # Get augmentend image samples
            orig = next(generator)

            # Get masks for each image sample
            if mask_init_seed is None:
                mask = np.stack([random_mask(orig.shape[1], orig.shape[2]) for _ in range(orig.shape[0])], axis=0)
            else:
                mask = np.stack([random_mask(orig.shape[1], orig.shape[2], mask_seed=mask_init_seed+seed_add+seed) for seed in range(orig.shape[0])], axis=0)
                seed_add = (seed_add + orig.shape[0]) % total_steps
            
            # Apply masks to all image sample
            masked = orig.copy()
            masked *= mask
                        
            # Yield ((img, mask),  label) training batches
            #gc.collect()
            yield [masked, mask], orig    


def random_mask(height, width, channels=3, mask_seed=None):
    """
    Generates a random irregular mask with lines, circles and elipses using OpenCV.

    It is a modified version of random_mask from 
    https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/util.py. 
    Modifications:
        1) Unnecessary conversion '1-img' was removed, np.ones are initialised at the 
        beginning instead of np.zeros.
        2) Reproducible masks have been implemented by passing 'mask_seed'.
    """

    if mask_seed is not None:
        seed(mask_seed)
    
    img = np.ones((height, width, channels), np.uint8)

    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")
    
    # Draw random lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img, (x1,y1), (x2,y2), (0,0,0), thickness)
        
    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img, (x1,y1), radius, (0,0,0), -1)
        
    # Draw random ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(0,0,0), thickness)
    
    return img


def torch_vgg_pp(x):
    """
    Image pre-processing function used in PyTorch.

    For an RGB image:
       1) 1/255 scaling,
       2) per-channel mean subtraction (0.485, 0.456, 0.406),
       2) per-channel scaling by std (0.229, 0.224, 0.225).
    """
    x /= 255.
    x[..., 0] -= 0.485
    x[..., 1] -= 0.456
    x[..., 2] -= 0.406
    x[..., 0] /= 0.229
    x[..., 1] /= 0.224
    x[..., 2] /= 0.225
    return x
