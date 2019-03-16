from keras import backend as K
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras.layers import Conv2D


class myPConv2D(Conv2D):
  
    def __init__(self, *args, last_layer=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_layer = last_layer
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """
        Adapted from original _Conv() layer of Keras.
        Parameters
            input_shape: list of dimensions for [img, mask].
        """
        assert isinstance(input_shape, list)
        assert self.data_format == 'channels_last', "data format should be `channels_last`"
        channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        
        self.input_dim = input_shape[0][channel_axis]
        
        # Image kernel:
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel  = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Image bias:
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Mask kernel:
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        self.built = True

    def call(self, inputs):
        assert isinstance(inputs, list) and len(inputs) == 2

        #img, mask = inputs
     
        # Masked convolution:
        img_output = K.conv2d(inputs[0] * inputs[1],
                              self.kernel, 
                              strides=self.strides, 
                              padding=self.padding, 
                              data_format=self.data_format)
      
        # Image scaling:
        # --------------
        sum_m = K.conv2d(inputs[1], 
                         self.kernel_mask, 
                         strides=self.strides, 
                         padding=self.padding, 
                         data_format=self.data_format)    
        # Note, sum_1 does not need to be created via conv2d (as sum_m), it can be generated straight away:
        sum_1i = self.kernel_size[0] * self.kernel_size[1] * self.input_dim
        sum_1 = sum_1i * K.ones(K.shape(sum_m))
        # Prevent division by zero:
        sum_m_clip = K.clip(sum_m, 1., None)
        # Scale the image:
        img_output = img_output * (sum_1 / sum_m_clip)
        
        # Apply bias only to the image (if chosen to do so):
        if self.use_bias:
            img_output = K.bias_add(img_output,
                                    self.bias,
                                    data_format=self.data_format)

        # Apply activation if needed. Note, in the paper, activation is applied after BatchNormalization.
        if self.activation is not None:
            img_output = self.activation(img_output)
    
        if self.last_layer:
            return img_output

        # Update the mask:
        mask_output = K.clip(sum_m, 0., 1.)
    
        return [img_output, mask_output]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert self.data_format == 'channels_last'
        space = input_shape[0][1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
        if self.last_layer:
            return new_shape
        return [new_shape, new_shape]
    
