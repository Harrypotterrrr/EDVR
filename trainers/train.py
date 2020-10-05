import os
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from utils.utils import ColorPrint as cp

class Trainer:
    def __init__(self, config, model, dataloader):
        self.config = config

        self.pretrained_model = self.config["pretrained_model"]
        self.version = self.config["version"]

        # training setting
        self.batch_size = self.config["batch_size"]
        self.num_epoch = self.config["num_epoch"]
        self.lr = self.config["lr"]
        self.lr_schr = self.config["lr_schr"]
        self.beta1 = self.config["beta1"]
        self.beta2 = self.config["beta2"]

        self.total_sample = self.config["total_sample"]
        self.num_step = (self.total_sample + self.batch_size - 1) // self.batch_size

        self.log_step = self.config["log_step"]
        self.log_epoch = self.config["log_epoch"]
        self.log_block_size = self.config["log_block_size"]
        self.val_step = self.config["val_step"]
        self.model_save_step = self.config["model_save_step"]

        # path
        self.log_path = self.config["log_path"]
        self.log_train_path = os.path.join(self.log_path, config["log_train_path"])
        self.log_val_path = os.path.join(self.log_path, config["log_val_path"])
        self.model_save_path = os.path.join(self.config["model_save_path"], self.config["version"])

        self.log_template = "Epoch: [%d/%d], Step: [%d/%d], time: %s/%s, loss: %.7f, psnr: %.4f, lr: %.2e"
        self.val_template = "Validation, val_loss: %.7f, val_psnr: %.4f"
        self.total_time = 0

        learning_rate_fn = self.init_lr()

        if self.config['gpus']:
            gpu_list = [f"/gpu:{i}" for i in range(len(self.config['gpus']))]
            cp.print_message(f'Distributing the model to GPU:{self.config["gpus"]} for training...')
            self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpu_list,
                                                               cross_device_ops=tf.distribute.NcclAllReduce())
            with self.mirrored_strategy.scope():
                self.model = model(config)
                self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=self.beta1, beta_2=self.beta2)
                self.train_dataset, self.val_dataset, self.test_dataset = dataloader()
                self.train_dataset = self.mirrored_strategy.experimental_distribute_dataset(self.train_dataset)
                self.val_dataset = self.mirrored_strategy.experimental_distribute_dataset(self.val_dataset)
                self.val_dataset = iter(self.val_dataset)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.model_save_path, max_to_keep=3)

        if self.pretrained_model is True: # TODO load step
            self.manager.restore_or_initialize()
            # print(self.manager.checkpoints) # checkpoints list
            # self.checkpoint.restore(tf.train.latest_checkpoint(self.model_save_path))
            cp.print_success(f"Restored from {self.manager.latest_checkpoint}")

        self.train_summary_writer = tf.summary.create_file_writer(self.log_train_path)
        self.val_summary_writer = tf.summary.create_file_writer(self.log_val_path)

    def write_log(self, writer, loss, psnr, logging):
        with writer.as_default():
            block_id = self.total_step // self.log_block_size
            start_step, end_step = block_id * self.log_block_size, (block_id + 1) * self.log_block_size
            tf.summary.scalar(f"loss/{self.version}", loss, step=self.total_step)
            tf.summary.scalar(f"psnr/{self.version}", psnr, step=self.total_step)
            tf.summary.text(f"{self.version}/Step {start_step}~{end_step}", logging, step=self.total_step)

    def init_lr(self):
        if self.lr_schr == "const":
            learning_rate_fn = self.lr
        elif self.lr_schr == "exp":
            learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                self.lr,
                decay_steps=self.config["lr_exp_step"],
                decay_rate=self.config["lr_exp_decay"],
                staircase=True
            )
        elif self.lr_schr == "step":
            learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=self.config["lr_boundary"],
                values=self.config["lr_boundary_value"],
            )

        return learning_rate_fn

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

    @tf.function
    def multi_validate_step(self, x, y):
        loss, psnr = self.mirrored_strategy.run(self.validate_step, args=(x, y))
        mean_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        mean_psnr = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, psnr, axis=None)
        return mean_loss, mean_psnr

    @tf.function
    def multi_train_step(self, x, y):
        loss, psnr = self.mirrored_strategy.run(self.train_step, args=(x, y))
        mean_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        mean_psnr = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, psnr, axis=None)
        return mean_loss, mean_psnr

    def train_epoch(self, epoch):

        loss_list = []
        psnr_list = []
        cp.print_message("Start training...")
        self.start_time = time.time()
        with self.mirrored_strategy.scope():
            for step, (batch_x, batch_y) in enumerate(self.train_dataset):
                self.checkpoint.step.assign_add(1)
                self.total_step += 1
                # loss, psnr = self.train_step(batch_x, batch_y)
                loss, psnr = self.multi_train_step(batch_x, batch_y)

                # values = [('train_loss',train_loss), ('train_acc'), train_acc]
                # self.progbar.update(step * self.batch_size, values=values)

                loss_list.append(loss)
                psnr_list.append(psnr)

                if step % self.log_step == 0:
                    elapsed, total_time = self.calc_time()
                    current_lr = self.optimizer._decayed_lr(tf.float32)
                    logging = self.log_template % (epoch, self.num_epoch, step, self.num_step, elapsed, total_time, loss, psnr, current_lr)
                    print(logging)
                    self.write_log(self.train_summary_writer, loss, psnr, logging)

                if step % self.val_step == 0:
                    val_batch_x, val_batch_y = self.val_dataset.get_next()
                    val_loss, val_psnr = self.multi_validate_step(val_batch_x, val_batch_y)

                    logging = self.val_template % (val_loss, val_psnr)
                    cp.print_success(logging)
                    self.write_log(self.val_summary_writer, val_loss, val_psnr, logging)

                if step % self.model_save_step == 0:
                    self.manager.save()  # save checkpoint

        loss = np.mean(loss_list)
        psnr = np.mean(psnr_list)
        return loss, psnr

    def train(self):
        # self.progbar = Progbar(target=self.config["total_sample"], interval=self.config["log_sec"])
        self.total_step = 0
        for epoch in range(1, self.num_epoch + 1):
            # print("epoch {}/{}".format(epoch, self.num_epoch))
            loss, acc = self.train_epoch(epoch)

            if epoch % self.config["log_epoch"] == 0:
                pass
                # TODO: Using logger instead of print function
                # print(template.format(epoch, loss, acc))
