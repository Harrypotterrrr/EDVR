import tensorflow as tf
from tensorflow import keras

from deformable_conv import deform_layer

class PCD_Align(object):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
        with 3 pyramid levels.
    '''
    def __init__(self, nf=64, groups=8):
        # L3: level3, 1/4 spatial size
        self.L3_offset_conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")  # concat for diff
        self.L3_offset_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.L3_dcnpack = deform_layer.DCN_seq(nf)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")  # concat for diff
        self.L2_offset_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")  # concat for offset
        self.L2_offset_conv3 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        self.L2_dcnpack = deform_layer.DCN_seq(nf)
        self.L2_fea_conv = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same") # concat for diff
        self.L1_offset_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same") # concat for offset
        self.L1_offset_conv3 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.L1_dcnpack = deform_layer.DCN_seq(nf)
        self.L1_fea_conv = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        # Cascading DCN
        self.cas_offset_conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.cas_offset_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.cas_dcnpack = deform_layer.DCN_seq(nf)
        self.lrelu = keras.layers.LeakyReLU(0.1)

    def __call__(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,H,W,C] features
        '''
        # L3
        L3_offset = tf.concat([nbr_fea_l[2], ref_fea_l[2]], axis=3)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L2_offset = tf.concat([nbr_fea_l[1], ref_fea_l[1]], axis=3)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset_shape = tf.shape(L3_offset)
        L3_offset = tf.image.resize(L3_offset, (2*L3_offset_shape[1], 2*L3_offset_shape[2]))
        L2_offset = self.lrelu(self.L2_offset_conv2(tf.concat([L2_offset, L3_offset * 2], axis=3)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea_shape = tf.shape(L3_fea)
        L3_fea = tf.image.resize(L3_fea, (2*L3_fea_shape[1], 2*L3_fea_shape[2]))
        L2_fea = self.lrelu(self.L2_fea_conv(tf.concat([L2_fea, L3_fea], axis=3)))
        # L1
        L1_offset = tf.concat([nbr_fea_l[0], ref_fea_l[0]], axis=3)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset_shape = tf.shape(L2_offset)
        L2_offset = tf.image.resize(L2_offset, (2 * L2_offset_shape[1], 2 * L2_offset_shape[2]))
        L1_offset = self.lrelu(self.L1_offset_conv2(tf.concat([L1_offset, L2_offset*2], axis=3)))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea_shape = tf.shape(L2_fea)
        L2_fea = tf.image.resize(L2_fea, (2 * L2_fea_shape[1], 2 * L2_fea_shape[2]))
        L1_fea = self.L1_fea_conv(tf.concat([L1_fea, L2_fea], axis=3))
        # Cascading
        offset = tf.concat([L1_fea, ref_fea_l[0]], axis=3)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))
        return L1_fea
