import os
import pathlib
import numpy as np
import tensorflow as tf


class REDSDataLoader:

    def __init__(self, config):

        self.config = config
        self.nframes = config["nframes"]
        self.center = self.nframes // 2
        self.batch_size = config["batch_size"]
        self.buffer_size = config["buffer_size"]
        self.prefetch_buffer_size = config["prefetch_buffer_size"]

        self.root_path = config["root_path"]
        self.train_path = os.path.join(self.root_path, 'train')
        self.val_path = os.path.join(self.root_path, 'val')
        self.test_path = os.path.join(self.root_path, 'test')

        self.tfrecord_path = config["tfrecord_path"]

        x_paths, y_paths = self.build_image_paths()
        self.config['total_sample'] = len(x_paths)
        if not os.path.exists(self.tfrecord_path):
            self.build_tfrecord(x_paths, y_paths)

    def build_image_paths(self): # train_blur_bicubic robustness coding TODO

        train_path = pathlib.Path(self.train_path)
        # self.train_path = root_path.joinpath('train')

        train_blur_bicubic_paths = []
        train_blur_bicubic_subpath = list(sorted(train_path.glob(f'{self.config["train_blur_bicubic"]}*')))
        for subpath in train_blur_bicubic_subpath:
            label = pathlib.Path(subpath).name
            train_blur_bicubic_subpath = [str(path) for path in train_path.glob(f'{self.config["train_blur_bicubic"]}{label}/*')]
            train_blur_bicubic_subpath = self.build_clips(sorted(train_blur_bicubic_subpath))
            train_blur_bicubic_paths.extend(train_blur_bicubic_subpath)

        train_blur_paths = list(sorted(train_path.glob(f'{self.config["train_blur"]}*/*')))
        train_blur_paths = [str(path) for path in train_blur_paths]

        train_blur_bicubic_ctr = len(train_blur_bicubic_paths)
        train_blur_ctr = len(train_blur_paths)

        # print(train_blur_ctr, train_blur_bicubic_ctr)
        assert train_blur_ctr == train_blur_bicubic_ctr
        return train_blur_bicubic_paths, train_blur_paths

    def build_dataloader(self, x_paths, y_paths): # deprecated
        self.dataset = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))

        def convert_types(image):
            image = tf.cast(image, tf.float32)
            image /= 255
            return image

        def load_image(x_path, y_path):
            x_image = []
            for i in x_path:
                x = tf.io.read_file(i)
                x_image.append(tf.image.decode_image(x, channels=3))
            x_image = tf.stack(x_image, dtype=tf.float32)

            y = tf.io.read_file(y_path)
            y_image = tf.image.decode_image(y, channels=3)

            return x_image, y_image

        self.dataset = self.dataset.shuffle(len(self.dataset))
        self.dataset = self.dataset.map(load_image).batch(self.batch_size)  # .repeat(10)

    def build_tfrecord(self, x_paths, y_paths):

        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for x_path, y_path in zip(x_paths, y_paths):
                y_path = open(y_path, 'rb').read()
                feature = {
                    'nframes': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.nframes])),
                    'target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_path]))
                }

                for i, _x in enumerate(x_path):
                    frame = open(_x, 'rb').read()
                    feature[f'frame{i}'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[frame]))

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()

    def build_clips(self, image_list: list):
        image_list = np.asarray(image_list)
        X_trn = []
        for i in range(len(image_list)):
            next_frames = image_list[i:i + self.center + 1]
            if i < self.center:
                prev_frames = image_list[:i]
            else:
                prev_frames = image_list[i - self.center:i]

            to_fill = self.nframes - next_frames.shape[0] - prev_frames.shape[0]
            if to_fill:
                if len(prev_frames) and i < self.nframes:
                    pad_x = prev_frames[0]
                    pad_x = np.repeat(pad_x, to_fill, axis=0)
                    xx = np.concatenate((pad_x, prev_frames, next_frames))
                else:
                    if i > self.nframes:
                        pad_x = next_frames[-1]
                        pad_x = np.repeat(pad_x, to_fill, axis=0)
                        xx = np.concatenate((prev_frames, next_frames, pad_x))
                    else:
                        pad_x = next_frames[0]
                        pad_x = np.repeat(pad_x, to_fill, axis=0)
                        xx = np.concatenate((pad_x, prev_frames, next_frames))
            else:
                xx = np.concatenate((prev_frames, next_frames))
            X_trn.append(xx)
        X_trn = np.stack(X_trn)
        return X_trn

        # X_trn = []
        # for i in range(len(images)):
        #     next_frames = X[i:i + center + 1]
        #     if i < center:
        #         prev_frames = X[:i]
        #     else:
        #         prev_frames = X[i - center:i]
        #
        #     to_fill = nframes - next_frames.shape[0] - prev_frames.shape[0]
        #     if to_fill:
        #         if len(prev_frames) and i < nframes:
        #             pad_x = prev_frames[0][np.newaxis, :]
        #             pad_x = np.repeat(pad_x, to_fill, axis=0)
        #             xx = np.concatenate((pad_x, prev_frames, next_frames))
        #         else:
        #             if i > nframes:
        #                 pad_x = next_frames[-1][np.newaxis, :]
        #                 pad_x = np.repeat(pad_x, to_fill, axis=0)
        #                 xx = np.concatenate((prev_frames, next_frames, pad_x))
        #             else:
        #                 pad_x = next_frames[0][np.newaxis, :]
        #                 pad_x = np.repeat(pad_x, to_fill, axis=0)
        #                 xx = np.concatenate((pad_x, prev_frames, next_frames))
        #     else:
        #         xx = np.concatenate((prev_frames, next_frames))
        #     X_trn.append(xx)
        # X_trn = np.stack(X_trn)

    def __call__(self):
        # self.preprocess()
        self.decode_tfrecord()
        return self.dataset

    def decode_tfrecord(self):
        self.dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        feature_description = {
            'nframes': tf.io.FixedLenFeature([], tf.int64),
            'target': tf.io.FixedLenFeature([], tf.string)
        }
        for i in range(self.nframes):
            feature_description[f'frame{i}'] = tf.io.FixedLenFeature([], tf.string)

        def _load_image(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            images = []
            for i in range(self.nframes):
                image = tf.io.decode_png(feature_dict[f'frame{i}'], channels=3)
                image = tf.cast(image, dtype='float32') / 255.
                images.append(image)
            images = tf.stack(images)
            target = tf.io.decode_png(feature_dict['target'], channels=3)
            target = tf.cast(target, dtype='float32') / 255.
            return images, target

        self.dataset = self.dataset.shuffle(buffer_size=self.buffer_size)
        self.dataset = self.dataset.map(_load_image)
        self.dataset = self.dataset.batch(batch_size=self.batch_size)
        self.dataset = self.dataset.prefetch(buffer_size=self.prefetch_buffer_size)

    def preprocess(self):
        self.train_data = self.train_data.map(
            REDSDataLoader.convert_types
        ).batch(self.config.batch_size)
        self.test_data = self.test_data.map(
            REDSDataLoader.convert_types
        ).batch(self.config.batch_size)

    @staticmethod
    def convert_types(batch):
        image, label = batch.values()
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

if __name__ == "__main__":

    from utils.config import process_config
    config = process_config()
    config["tfrecord_path"] = "../tfrecord"

    from models.EDVR import EDVR
    x = tf.ones(shape=[4, 5, 64, 64, 3])
    model = EDVR(config)

    dataloader = REDSDataLoader(config)
    for step, (inputs, targets) in enumerate(dataloader()):
        print(inputs.shape)
        print(targets.shape)
        with tf.GradientTape() as tape:
            y = model(inputs)
        print(y.shape)
