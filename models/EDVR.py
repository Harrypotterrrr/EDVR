import os
import functools
import tensorflow as tf
from tensorflow import keras

from models import module_util
from models.PCD import PCD_Align
from models.TSA import TSA_Fusion
from models.Predeblur import Predeblur_ResNet_Pyramid

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# logging.disable(logging.WARNING)
# tf.get_logger().setLevel('INFO')


class EDVR(object):

    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10,
                 center=None, predeblur=False, HR_in=False, w_TSA=True):
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(module_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        else:
            if self.HR_in:
                self.conv_first_1 = keras.layers.Conv2D(nf, (3, 3), strides=(1, 1), padding="same", use_bias=True)
                self.conv_first_2 = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same", use_bias=True)
                self.conv_first_3 = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same", use_bias=True)
            else:
                self.conv_first = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same", use_bias=True)
        self.feature_extraction = module_util.Module(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = keras.layers.Conv2D(nf, (3, 3), (2, 2), "same")
        self.fea_L2_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        self.fea_L3_conv1 = keras.layers.Conv2D(nf, (3, 3), (2, 2), "same")
        self.fea_L3_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nf, center=self.center)
        else:
            self.tsa_fusion = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        #### reconstruction
        self.recon_trunk = module_util.Module(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = keras.layers.Conv2D(nf * 4, (3, 3), (1, 1), "same")
        self.upconv2 = keras.layers.Conv2D(64 * 4, (3, 3), (1, 1), "same")
        self.pixel_shuffle = lambda x:tf.nn.depth_to_space(x, 2)
        self.HRconv = keras.layers.Conv2D(64, (3, 3), (1, 1), "same")
        self.conv_last = keras.layers.Conv2D(3, (3, 3), (1, 1), "same")
        #### activation function
        self.lrelu = keras.layers.LeakyReLU(0.1)

    def __call__(self, x):
        x_shape = tf.shape(x)
        B = x_shape[0]
        N = x_shape[1]
        H = x_shape[2]
        W = x_shape[3]
        C = x_shape[4]
        x_center = tf.cast(x[:, self.center, :, :, :], tf.float32)

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(tf.reshape(x, [-1, H, W, C]))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H = tf.divide(H, 4)
                W = tf.divide(W, 4)
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(tf.reshape(x, [-1, H, W, C])))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H = tf.divide(H, 4)
                W = tf.divide(W, 4)
            else:
                L1_fea = self.lrelu(self.conv_first(tf.reshape(x, [-1, H, W, C])))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = tf.reshape(L1_fea, [B, N, H, W, -1])
        L2_fea = tf.reshape(L2_fea, [B, N, H // 2, W // 2, -1])
        L3_fea = tf.reshape(L3_fea, [B, N, H // 4, W // 4, -1])

        #### pcd align
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :], L2_fea[:, self.center, :, :, :],
            L3_fea[:, self.center, :, :, :]
        ]
        aligned_fea = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        def cond(i, N, fea_col):
            return i < N
        def body(i, N, fea_col):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :], L2_fea[:, i, :, :, :],
                L3_fea[:, i, :, :, :]
            ]
            fea_col = fea_col.write(i, self.pcd_align(nbr_fea_l, ref_fea_l))
            i = tf.add(i, 1)
            return i, N, fea_col
        _, _, aligned_fea = tf.while_loop(cond, body, [0, N, aligned_fea])
        aligned_fea = aligned_fea.stack()
        aligned_fea = tf.transpose(aligned_fea, [1, 0, 2, 3, 4])  # [B, N, H, W, C]
        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        if self.HR_in:
            base = x_center
        else:
            x_center_shape = tf.shape(x_center)
            base = tf.image.resize(x_center, [4 * x_center_shape[1], 4 * x_center_shape[2]])
        out = tf.add(out, base)
        return out

    @staticmethod
    def l1_loss(x, y, eps=1e-6):
        diff = x - y
        loss = tf.reduce_sum(tf.sqrt(diff * diff + eps))
        return loss


if __name__ == "__main__":

    # tf.enable_eager_execution()
    x = tf.ones(shape=[4, 5, 64, 64, 3])
    model = EDVR()
    print(model(x))
