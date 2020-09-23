import tensorflow as tf

from models import EDVR


def EDVR_model_fn(features, labels, mode, params):
    netG = EDVR.EDVR(params["nf"], params["nframes"],
                     params["groups"], params["front_RBs"],
                     params["back_RBs"], params["center"],
                     params["predeblur"], params["HR_in"],
                     params["w_TSA"])


class ExampleModel(tf.keras.Model):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    @staticmethod
    def loss_object(prediction, label):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        return loss_object(prediction, label)
