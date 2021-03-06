from functools import partial
import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate


def get_unet(hidden_activation='relu', initializer='he_normal', output_activation='sigmoid'):
    """Builds a Convolutional Neural Network for semantic segmentation 
    with an architecture based on U-Net. 

        Args:
            hidden_activation: The hidden activation function to use, 
            in the convolutional layers, defaults to ReLU.
            initializer: The kernel initializer to use in the convolutional layers,
            he_normal by default.
            output_activation: The output activation function, defaults to sigmoid.

        Returns:
            A Tensorflow Model with two paths, the first path is the encoder
            and is responsible for capturing the context in the image, it is
            a stack of three convolutional layers with Max Pooling.

            The second path forms the decoder and has three upsampling convlutional
            layers to expand the low resolution features from the encoder into a 
            higher resolution for prediction.

            The two paths are connected by two central convolutional layers. 
    """
    PartialConv = partial(Conv2D,
     activation=hidden_activation,
     kernel_initializer=initializer,      
     padding='same')
    
    #ENCODER
    model_input = Input(shape=(None, None, 3))
    enc_cov_1 = PartialConv(32, 3)(model_input)
    enc_cov_1 = PartialConv(32, 3)(enc_cov_1)
    enc_pool_1 = MaxPooling2D(pool_size=(2, 2))(enc_cov_1)
    
    enc_cov_2 = PartialConv(64, 3)(enc_pool_1)
    enc_cov_2 = PartialConv(64, 3)(enc_cov_2)
    enc_pool_2 = MaxPooling2D(pool_size=(2, 2))(enc_cov_2)
    
    enc_cov_3 = PartialConv(128, 3)(enc_pool_2)
    enc_cov_3 = PartialConv(128, 3)(enc_cov_3)
    enc_pool_3 = MaxPooling2D(pool_size=(2, 2))(enc_cov_3)
    
    #CENTER
    center_cov = PartialConv(256, 3)(enc_pool_3)
    center_cov = PartialConv(256, 3)(center_cov)
    
    #DECODER
    upsampling1 = UpSampling2D(size=(2, 2))(center_cov)
    dec_up_conv_1 = PartialConv(128, 2)(upsampling1)
    dec_merged_1 = Concatenate(axis=3)([enc_cov_3, dec_up_conv_1])
    dec_conv_1 = PartialConv(128, 3)(dec_merged_1)
    dec_conv_1 = PartialConv(128, 3)(dec_conv_1)
    
    upsampling2 = UpSampling2D(size=(2, 2))(dec_conv_1)
    dec_up_conv_2 = PartialConv(64, 2)(upsampling2)
    dec_merged_2 = Concatenate(axis=3)([enc_cov_2, dec_up_conv_2])
    dec_conv_2 = PartialConv(64, 3)(dec_merged_2)
    dec_conv_2 = PartialConv(64, 3)(dec_conv_2)
    
    upsampling3 = UpSampling2D(size=(2, 2))(dec_conv_2)
    dec_up_conv_3 = PartialConv(32, 2)(upsampling3)
    dec_merged_3 = Concatenate(axis=3)([enc_cov_1, dec_up_conv_3])
    dec_conv_3 = PartialConv(32, 3)(dec_merged_3)
    dec_conv_3 =  PartialConv(32, 3)(dec_conv_3)
    
    #OUTPUT
    output = Conv2D(1, 1, activation=output_activation)(dec_conv_3)
    
    return tf.keras.Model(inputs=model_input, outputs=output)