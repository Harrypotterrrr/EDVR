import tensorflow as tf
from tensorflow import keras

from models import module_util


class Predeblur_ResNet_Pyramid(object):
    def __init__(self, nf=128, HR_in=False):
        """
        :param nf: number of filters
        :param HR_in: True if the inputs are high spatial size
        """
        self.HR_in = True if HR_in else False
        if self.HR_in:
            self.conv_first_1 = keras.layers.Conv2D(nf, (3, 3), strides=(1, 1), padding="same", use_bias=True)
            self.conv_first_2 = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same", use_bias=True)
            self.conv_first_3 = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same", use_bias=True)
        else:
            self.conv_first = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same", use_bias=True)
        self.RB_L1_1 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L1_2 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L1_3 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L1_4 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L1_5 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L2_1 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L2_2 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L3_1 = module_util.ResidualBlock_noBN(nf=nf)
        self.deblur_L2_conv = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same")
        self.deblur_L3_conv = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same")
        self.lrelu = keras.layers.LeakyReLU(0.1)

    def __call__(self, x):
        if self.HR_in:
            L1_fea = self.lrelu(self.conv_first_1(x))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = self.RB_L3_1(L3_fea)
        L3_fea_shape = tf.shape(L3_fea)
        L3_fea = tf.image.resize_images(L3_fea, (2 * L3_fea_shape[1], 2 * L3_fea_shape[2]))

        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = self.RB_L2_2(L2_fea)
        L2_fea_shape = tf.shape(L2_fea)
        L2_fea = tf.image.resize_images(L2_fea, (2 * L2_fea_shape[1], 2 * L2_fea_shape[2]))
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out
