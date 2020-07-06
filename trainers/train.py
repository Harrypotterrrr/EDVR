import tensorflow as tf
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.optimizer = tf.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # multiple GPU
        visible_gpus = tf.config.experimental.list_physical_devices('GPU')
        print('Visible devices : ', visible_gpus)
        gpu_ids = [0]
        devices = ['/device:GPU:{}'.format(i) for i in gpu_ids]
        self.strategy = tf.distribute.MirroredStrategy(devices=devices)

        # Global batch size
        BATCH_SIZE_PER_REPLICA = 2
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync
        # Buffer size for data loader
        BUFFER_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync * 16

        self.data = self.strategy.experimental_distribute_dataset(self.data)

    @tf.function
    def train_step(self, x, y):
        with self.strategy.scope():
            with tf.GradientTape() as tape:
                predictions = self.model(x)
                # loss = self.model.loss_object(y, predictions)
                loss = self.model.l1_loss(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(y, predictions)

        return self.train_loss.result(), self.train_accuracy.result()

    def train_epoch(self):
        with self.strategy.scope():
            loop = tqdm(range(self.config.num_iter_per_epoch))
            losses = []
            accs = []
            for _, (batch_x, batch_y) in zip(loop, self.data):
                train_loss, train_accuracy = self.train_step(batch_x, batch_y)
                losses.append(train_loss)
                accs.append(train_accuracy)
            loss = np.mean(losses)
            acc = np.mean(accs)
        return loss, acc

    def train(self):
        epochs = self.config.num_epochs
        for epoch in range(1, epochs+1):
            loss, acc = self.train_epoch()
            template = "Epoch: {} | Train Loss: {}, Train Accuracy: {}"
            if epoch % self.config.verbose_epochs == 0:
                # TODO: Using logger instead of print function
                print(template.format(epoch, loss, acc))
