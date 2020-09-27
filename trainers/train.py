import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar


class Trainer:
    def __init__(self, config, model, dataloader):
        self.config = config

        self.pretrained_model = self.config["pretrained_model"]

        self.batch_size = self.config["batch_size"]
        self.num_epoch = self.config["num_epoch"]
        self.lr = self.config["lr"]

        self.total_sample = self.config["total_sample"]
        self.num_step = (self.total_sample + self.batch_size - 1) // self.batch_size

        self.log_epoch = self.config["log_epoch"]
        self.sample_epoch = self.config["sample_epoch"]
        self.model_save_epoch = self.config["model_save_epoch"]

        self.model_save_path = self.config["model_save_path"]


        self.log_template = "Epoch: [%d/%d], Step: [%d/%d], time: %s, loss: %.7f, psnr: %.4f, lr: %.2e"

        self.model = model(config)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.config["lr"])
        self.dataset = dataloader()

    def calc_psnr(self, pred, target, max_val=1.0): # TODO max_val
        psnr = tf.image.psnr(pred, target, max_val=max_val)
        return tf.reduce_mean(psnr)

    def calc_time(self, start_time):
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        start_time = time.time()

        return elapsed, start_time

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.model.loss_object(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        psnr = self.calc_psnr(y, predictions)
        return loss, psnr

    def train_epoch(self, epoch):

        loss_list = []
        psnr_list = []
        start_time = time.time()
        for step, (batch_x, batch_y) in enumerate(self.dataset):
            loss, psnr = self.train_step(batch_x, batch_y)

            elapsed, start_time = self.calc_time(start_time)

            print(self.log_template % (epoch, self.num_epoch, step, self.num_step, elapsed, loss, psnr, self.lr))

            # values = [('train_loss',train_loss), ('train_acc'), train_acc]
            # self.progbar.update(step * self.batch_size, values=values)

            loss_list.append(loss)
            psnr_list.append(psnr)

        loss = np.mean(loss_list)
        psnr = np.mean(psnr_list)
        return loss, psnr

    def train(self):
        # self.progbar = Progbar(target=self.config["total_sample"], interval=self.config["log_sec"])

        for epoch in range(1, self.num_epoch + 1):
            # print("epoch {}/{}".format(epoch, self.num_epoch))
            loss, acc = self.train_epoch(epoch)

            if epoch % self.config["log_epoch"] == 0:
                pass
                # TODO: Using logger instead of print function
                # print(template.format(epoch, loss, acc))
