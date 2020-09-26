import tensorflow as tf
from tensorflow import keras


class TSA_Fusion(tf.keras.Model):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''
    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()

        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        self.tAtt_2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        # spatial attention (after fusion conv)
        self.sAtt_1 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.maxpool = keras.layers.MaxPool2D((3, 3), (2, 2), padding="same")
        self.avgpool = keras.layers.AveragePooling2D((3, 3), (2, 2), padding="same")
        self.sAtt_2 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.sAtt_3 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.sAtt_4 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.sAtt_5 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.sAtt_L1 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.sAtt_L2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.sAtt_L3 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.sAtt_add_1 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.sAtt_add_2 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.lrelu = keras.layers.LeakyReLU(0.1)

    def __call__(self, aligned_fea):

        aligned_fea_shape = tf.shape(aligned_fea)
        B = aligned_fea_shape[0]
        N = aligned_fea_shape[1]
        H = aligned_fea_shape[2]
        W = aligned_fea_shape[3]
        C = aligned_fea_shape[4]
        print("aligned_feature.shape:", aligned_fea.shape)
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :])
        emb = tf.reshape(self.tAtt_1(tf.reshape(aligned_fea, [-1, H, W, C])), [B, N, H, W, -1])
        cor_l = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        def cond(i, N, input, arr):
            return tf.less(i, N)
        def body(i, N, input, arr):
            emb_nbr = input[:, i, :, :, :]
            cor_tmp = tf.reduce_sum(emb_nbr * emb_ref, axis=3) # B, H, W
            arr = arr.write(i, cor_tmp)
            i = tf.add(i, 1)
            return i, N, input, arr
        _, _, _, cor_l = tf.while_loop(cond, body, [0, N, emb, cor_l])
        cor_l = cor_l.stack()
        cor_l = tf.transpose(cor_l, [1, 0, 2, 3])
        cor_prob = tf.sigmoid(cor_l)  # B, N, H, W
        cor_prob = tf.expand_dims(cor_prob, axis=4)
        cor_prob = tf.tile(cor_prob, [1, 1, 1, 1, C])
        cor_prob = tf.reshape(cor_prob, [B, H, W, -1])
        aligned_fea = tf.reshape(aligned_fea, [B, H, W, -1]) * cor_prob
        ### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))
        ### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(tf.concat([att_max, att_avg], axis=3)))
        ### pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(tf.concat([att_max, att_avg], axis=3)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L_shape = tf.shape(att_L)
        att_L = tf.image.resize(att_L, [2 * att_L_shape[1], 2 * att_L_shape[2]])
        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att_shape = tf.shape(att)
        att = tf.image.resize(att, [2 * att_shape[1], 2 * att_shape[2]])
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = tf.sigmoid(att)
        fea = fea * att * 2 + att_add
        return fea
