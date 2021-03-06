from keras import backend as K
from keras.applications.vgg16 import VGG16 
from keras.models import Model


def vgg16_feature_model(flayers, weights='imagenet'):
    """
    Feature exctraction VGG16 model.
  
    # Arguments
        flayers: list of strings with names of layers to get the features for. 
            The length of `flayers` should be > 1, otherwise the output shape 
            is one axis less.
        weights: ether "imagenet" or path to the file with weights. 
    # Returns
        features_model: keras.models.Model instance to extract the features.
      
    # Raises
        AssertionError: in case of `flayers` is not a list.
        AssertionError: in case of length of 'flayers' < 2.
    """
  
    assert isinstance(flayers,list), "First argument 'flayers' must be a list"
    assert len(flayers) > 1, "Length of 'flayers' must be > 1."
  
    base_model = VGG16(include_top=False, weights=weights)

    vgg16_outputs = [base_model.get_layer(flayers[i]).output for i in range(len(flayers))]

    features_model = Model(inputs=[base_model.input], outputs=vgg16_outputs, name='vgg16_features')
    features_model.trainable = False
    features_model.compile(loss='mse', optimizer='adam')
    
    return features_model

  
# Losses:
# -------
 
def total_loss(mask, vgg16_weights='imagenet'):
    """
    Total loss defined in Eq 7 of Liu et al 2018 with:
    y_true = I_gt,
    y_pred = I_out,
    y_comp = I_comp.
    """
    vgg16_lnames = ['block1_pool', 'block2_pool', 'block3_pool']
    vgg_model = vgg16_feature_model(vgg16_lnames, weights=vgg16_weights)
    def loss(y_true, y_pred):
        mask_inv = 1 - mask
        y_comp   = mask * y_true + mask_inv * y_pred
        vgg_out  = vgg_model(y_pred)
        vgg_gt   = vgg_model(y_true)
        vgg_comp = vgg_model(y_comp)
      
        l_valid = loss_per_pixel(y_true, y_pred, mask)
        l_hole  = loss_per_pixel(y_true, y_pred, mask_inv)
        l_perc  = loss_perc(vgg_out, vgg_gt, vgg_comp)
        l_style = loss_style(vgg_out, vgg_gt, vgg_comp) 
        l_tv    = loss_tv(y_comp, mask_inv)
    
        return l_valid + 6.*l_hole + 0.05*l_perc + 120.*l_style + 0.1*l_tv
    
    return loss


def loss_l1(y_true, y_pred):
    """ 
    Size-averaged L1 loss used in all the losses.
    
    If size_average is True, the l1 losses are means,
    If size_average is False, the l1 losses are sums divided by norm (should be specified), 
        only have effect if y_true.ndim = 4.
    """
    
    if K.ndim(y_true) == 4:
        # images and vgg features
        return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
    elif K.ndim(y_true) == 3:
        # gram matrices
        return K.mean(K.abs(y_pred - y_true), axis=[1,2])
    else:
        raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

    
def gram_matrix(x):
    """Gram matrix used in the style losses."""
    assert K.ndim(x) == 4, 'Input tensor should be 4D (B, H, W, C).'
    assert K.image_data_format() == 'channels_last', "Use channels-last format."

    # Permute channels and get resulting shape
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    
    shape = K.shape(x)
    B, C, H, W = shape[0], shape[1], shape[2], shape[3]
    
    # Reshape x and do batch dot product
    features = K.reshape(x, K.stack([B, C, H*W]))
    
    gram = K.batch_dot(features, features, axes=2)

    # Normalize with channels, height and width
    gram /= K.cast(C * H * W, x.dtype) 
    
    return gram


def loss_per_pixel(y_true, y_pred, mask):
    """
    Per pixel loss for selected pixels.
    Note, we don't use loss_l1 for l_valid and l_hole to prevent unnecessary 
        mask multiplication, "mask * y_pred - mask *y_true)" instead of 
        "mask * (y_pred - y_true)".
    """
    assert K.ndim(y_true) == 4, 'Input tensor should be 4D (B, H, W, C).'
    return K.mean(K.abs(mask * (y_pred - y_true)), axis=[1,2,3])
  
  
def loss_perc(vgg_out, vgg_gt, vgg_comp):
    """Perceptual loss."""
    l = 0.
    for o, g, c in zip(vgg_out, vgg_gt, vgg_comp):
        l += loss_l1(o, g) + loss_l1(c, g)
    return l

  
def loss_style(vgg_out, vgg_gt, vgg_comp):
    """Style loss consisting of two terms: out and comp."""
    l = 0.
    for o, g, c in zip(vgg_out, vgg_gt, vgg_comp):
        gram_gt = gram_matrix(g)
        l += loss_l1(gram_matrix(o), gram_gt) + loss_l1(gram_matrix(c), gram_gt)  
    return l
  

def loss_tv(y_comp, mask_inv):
    """Total variation (TV) loss, smoothing penalty on the hole region."""
    assert K.ndim(y_comp) == 4 and K.ndim(mask_inv) == 4, 'Input tensors should be 4D (B, H, W, C).'

    # Create dilated hole region using a 3x3 kernel of all 1s.
    kernel = K.ones(shape=(3, 3, mask_inv.shape[3], mask_inv.shape[3]))    

    dilated_mask = K.conv2d(mask_inv, kernel, data_format='channels_last', padding='same')
    dilated_mask = K.clip(dilated_mask, 0., 1.)

    #   It seems difficult to get the right value using the backend functions only.
    #   Our TV loss is higher than the one in the paper due to redundant 1-pixel margins
    #   and the way we implemented the TV loss via the conv2D operation. The offset depends 
    #   on the mask used and is constant for a given mask during optimization. This 
    #   additional constant will add noise to the loss values printed but should not affect 
    #   the gradients. The TV loss part based on the inpainted regions should be scaled 
    #   well with the weight 0.1 from the paper.

    # Compute dilated hole region of y_comp
    P = dilated_mask * y_comp
    
    return loss_l1(P[:,:-1,:,:], P[:,1:,:,:]) + loss_l1(P[:,:,:-1,:], P[:,:,1:,:])
