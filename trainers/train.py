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

        self.log_step = self.config["log_step"]
        self.log_epoch = self.config["log_epoch"]
        self.val_step = self.config["val_step"]
        self.model_save_step = self.config["model_save_step"]

        self.model_save_path = self.config["model_save_path"]


        self.log_template = "Epoch: [%d/%d], Step: [%d/%d], time: %s/%s, loss: %.7f, psnr: %.4f, lr: %.2e"
        self.val_template = "Validation, val_loss: %.7f, val_psnr: %.4f"
        self.total_time = 0

        if self.config['gpus']:
            self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],
                                                               cross_device_ops=tf.distribute.NcclAllReduce())
            with self.mirrored_strategy.scope():
                self.model = model(config)
                self.optimizer = tf.optimizers.Adam(learning_rate=self.config["lr"])
                self.train_dataset, self.val_dataset, self.test_dataset = dataloader()
                self.train_dataset = self.mirrored_strategy.experimental_distribute_dataset(self.train_dataset)
                self.val_dataset = self.mirrored_strategy.experimental_distribute_dataset(self.val_dataset)
                self.val_dataset = iter(self.val_dataset)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.model_save_path, max_to_keep=3)

        if self.pretrained_model is True:
            self.manager.restore_or_initialize()
            # print(self.manager.checkpoints) # checkpoints list
            # self.checkpoint.restore(tf.train.latest_checkpoint(self.model_save_path))
            print(f"Restored from {self.manager.latest_checkpoint}")

    def calc_psnr(self, pred, target, max_val=1.0): # TODO max_val
        psnr = tf.image.psnr(pred, target, max_val=max_val)
        return tf.reduce_mean(psnr)

    def calc_time(self):
        elapsed = time.time() - self.start_time
        self.total_time += elapsed
        elapsed = str(datetime.timedelta(seconds=elapsed))
        total_time = str(datetime.timedelta(seconds=int(self.total_time)))
        self.start_time = time.time()

        return elapsed, total_time

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.model.loss_object(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        psnr = self.calc_psnr(y, predictions)
        return loss, psnr

    @tf.function
    def validate_step(self, x, y):
        predictions = self.model(x)
        loss = self.model.loss_object(y, predictions)
        psnr = self.calc_psnr(y, predictions)
        return loss, psnr

    def multi_validate_step(self, x, y):
        loss, psnr = self.mirrored_strategy.run(self.validate_step, args=(x, y))
        mean_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        mean_psnr = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, psnr, axis=None)
        return mean_loss, mean_psnr

    def multi_train_step(self, x, y):
        loss, psnr = self.mirrored_strategy.run(self.train_step, args=(x, y))
        mean_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        mean_psnr = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, psnr, axis=None)
        return mean_loss, mean_psnr

    def train_epoch(self, epoch):

        loss_list = []
        psnr_list = []
        print("Start training...")
        self.start_time = time.time()
        with self.mirrored_strategy.scope():
            for step, (batch_x, batch_y) in enumerate(self.train_dataset):
                self.checkpoint.step.assign_add(1)
                # loss, psnr = self.train_step(batch_x, batch_y)
                loss, psnr = self.multi_train_step(batch_x, batch_y)

                elapsed, total_time = self.calc_time()

                # values = [('train_loss',train_loss), ('train_acc'), train_acc]
                # self.progbar.update(step * self.batch_size, values=values)

                loss_list.append(loss)
                psnr_list.append(psnr)

                if step % self.log_step == 0:
                    print(self.log_template % (epoch, self.num_epoch, step, self.num_step, elapsed, total_time, loss, psnr, self.lr))

                if step % self.val_step == 0: # TODO print color
                    val_batch_x, val_batch_y = self.val_dataset.get_next()
                    val_loss, val_psnr = self.multi_validate_step(val_batch_x, val_batch_y)
                    print(self.val_template % (val_loss, val_psnr))

                if step % self.model_save_step == 0:
                    self.manager.save()  # save checkpoint

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
