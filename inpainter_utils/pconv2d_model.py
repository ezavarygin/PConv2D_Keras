from .pconv2d_layer import myPConv2D
from .pconv2d_loss import total_loss
from keras.layers import BatchNormalization, Input
from keras.layers import ReLU, LeakyReLU, UpSampling2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam

def encoder_block(input_img, input_mask, filters, kernel_size, batch_norm=True, freeze_bn=False, activation=None, count=''):
    """
    Encoder block of layers.
    
    Parameters
        input_img: tensor, input image, output of an Input layer or a previous 
            encoder block.
        input_mask: tensor, input binary mask, output of an Input layer or a 
            previous encoder block.
        filters: integer, number of output channels.
        kernel_size: integer, width and height of the kernel.
        strides: integer, stride in both directions.
        batch_norm: boolean, whether to apply BatchNorm to the feature map 
            (before activation if applied).
        freeze_bn: boolean, whether to freeze the BatchNorm (fine-tuning stage).
        activation: boolean, whether to apply a ReLU activation at the end.
        count: string, block count to append to the end of layers' names.
      
    """
    if count != '':
        count = '_' + count
    
    pconv, mask = myPConv2D(filters,
                            kernel_size,
                            strides=2,
                            padding='same',
                            use_bias=not batch_norm,
                            kernel_initializer='he_uniform',
                            name='pconv2d_enc'+count
                           )([input_img, input_mask])

    if batch_norm:
        pconv = BatchNormalization(name='bn_enc'+count)(pconv, training=not freeze_bn)                     
    
    pconv = ReLU(name='relu'+count)(pconv)
    
    return pconv, mask

  
def decoder_block(prev_up_img, prev_up_mask, enc_img, enc_mask, filters, last_layer=False, count=''):
    """
    Decoder block of layers.
    
    Parameters
        prev_up_img: previous image layer to up-sample.
        prev_up_mask: previous mask layer to up-sample.
        enc_img: image from encoder stage to concatenate with up-sampled image.
        enc_mask: mask from encoder stage to concatenate with up-sampled mask.
        filters: integer, number of output channels in the PConv2D layer.
        count: string, block count to append to the end of layers' names.
        last_layer: boolean, whether this is the last decoder block (no mask will 
            be returned, no BatchNorm and no activation will be applied).
    """
    if count != '':
        count = '_' + count
    
    up_img  = UpSampling2D(size=2, name='img_upsamp_dec' + count)(prev_up_img)
    up_mask = UpSampling2D(size=2, name='mask_upsamp_dec' + count)(prev_up_mask)
    conc_img  = Concatenate(name='img_concat_dec' + count)([up_img, enc_img])
    conc_mask = Concatenate(name='mask_concat_dec' + count)([up_mask, enc_mask])

    if last_layer:
        return myPConv2D(filters, 3, strides=1, padding='same', use_bias=True, kernel_initializer='he_uniform', last_layer=last_layer, name='pconv2d_dec'+count)([conc_img, conc_mask])

    pconv, mask = myPConv2D(filters, 3, strides=1, padding='same', use_bias=False, kernel_initializer='he_uniform', name='pconv2d_dec'+count)([conc_img, conc_mask])
    pconv = BatchNormalization(name='bn_dec'+count)(pconv)
    pconv = LeakyReLU(alpha=0.2, name='leaky_dec'+count)(pconv)
    
    return pconv, mask


def pconv_model(fine_tuning=False, lr=0.0002, predict_only=False, image_size=(512, 512)):
    """Inpainting model."""

    img_input  = Input(shape=(image_size[0], image_size[1], 3), name='input_img')
    mask_input = Input(shape=(image_size[0], image_size[1], 3), name='input_mask')
    
    # Encoder:
    # --------
    e_img_1, e_mask_1 = encoder_block(img_input, mask_input, 64, 7, batch_norm=False, count='1')
    e_img_2, e_mask_2 = encoder_block(e_img_1, e_mask_1, 128, 5, freeze_bn=fine_tuning, count='2')
    e_img_3, e_mask_3 = encoder_block(e_img_2, e_mask_2, 256, 5, freeze_bn=fine_tuning, count='3')
    e_img_4, e_mask_4 = encoder_block(e_img_3, e_mask_3, 512, 3, freeze_bn=fine_tuning, count='4')
    e_img_5, e_mask_5 = encoder_block(e_img_4, e_mask_4, 512, 3, freeze_bn=fine_tuning, count='5')
    e_img_6, e_mask_6 = encoder_block(e_img_5, e_mask_5, 512, 3, freeze_bn=fine_tuning, count='6')
    e_img_7, e_mask_7 = encoder_block(e_img_6, e_mask_6, 512, 3, freeze_bn=fine_tuning, count='7')
    e_img_8, e_mask_8 = encoder_block(e_img_7, e_mask_7, 512, 3, freeze_bn=fine_tuning, count='8')

    # Decoder:
    # --------
    d_img_9, d_mask_9   = decoder_block(e_img_8, e_mask_8, e_img_7, e_mask_7, 512, count='9')
    d_img_10, d_mask_10 = decoder_block(d_img_9, d_mask_9, e_img_6, e_mask_6, 512, count='10')
    d_img_11, d_mask_11 = decoder_block(d_img_10, d_mask_10, e_img_5, e_mask_5, 512, count='11')
    d_img_12, d_mask_12 = decoder_block(d_img_11, d_mask_11, e_img_4, e_mask_4, 512, count='12')
    d_img_13, d_mask_13 = decoder_block(d_img_12, d_mask_12, e_img_3, e_mask_3, 256, count='13')
    d_img_14, d_mask_14 = decoder_block(d_img_13, d_mask_13, e_img_2, e_mask_2, 128, count='14')
    d_img_15, d_mask_15 = decoder_block(d_img_14, d_mask_14, e_img_1, e_mask_1, 64, count='15')
    d_img_16 = decoder_block(d_img_15, d_mask_15, img_input, mask_input, 3, last_layer=True, count='16')
    
    model = Model(inputs=[img_input, mask_input], outputs=d_img_16)

    # This will also freeze bn parameters `beta` and `gamma`: 
    #if fine_tuning:
    #    for l in model.layers:
    #        if 'bn_enc' in l.name:
    #            l.trainable = False
    
    if predict_only:
        return model
  
    model.compile(Adam(lr=lr), loss=total_loss(mask_input))
    
    return model
