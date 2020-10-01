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
        self.config["train_tfrecord_path"] = f"{self.tfrecord_path}_train"
        self.config["val_tfrecord_path"] = f"{self.tfrecord_path}_val"
        self.config["test_tfrecord_path"] = f"{self.tfrecord_path}_test"

        print("Start preparing images...")
        train_x_paths, train_y_paths = self.build_image_paths('train', 'sharp_bicubic', 'sharp') # write into config
        self.config['total_sample'] = len(train_x_paths)

        if not os.path.exists(f"{self.tfrecord_path}_train"):
            val_x_paths, val_y_paths = self.build_image_paths('val', 'sharp_bicubic', 'sharp')
            test_x_paths, _ = self.build_image_paths('test', 'sharp_bicubic')

            print("Start building train tfrecord...")
            self.build_tfrecord('train', train_x_paths, train_y_paths)
            print("Start building validation tfrecord...")
            self.build_tfrecord('val', val_x_paths, val_y_paths)
            print("Start building test tfrecord...")
            self.build_tfrecord('test', test_x_paths)

    def build_image_paths(self, type, x_tag, y_tag=""):

        if type == 'train':
            sub_path = pathlib.Path(self.train_path)
        elif type == 'val':
            sub_path = pathlib.Path(self.val_path)
        elif type == 'test':
            sub_path = pathlib.Path(self.test_path)
        # self.train_path = root_path.joinpath('train')

        x_dir = f"{type}_{x_tag}/X4/" if "bicubic" in x_tag else f"{type}_{x_tag}"
        y_dir = f"{type}_{y_tag}/"

        x_paths = []
        x_subpath = list(sorted(sub_path.glob(f'{x_dir}*')))
        for subpath in x_subpath:
            label = pathlib.Path(subpath).name
            x_subpath = [str(path) for path in sub_path.glob(f'{x_dir}{label}/*')]
            x_subpath = self.build_clips(sorted(x_subpath))
            x_paths.extend(x_subpath)

        x_ctr = len(x_paths)
        print(f"{type} input count: {x_ctr}")

        if type == "test":
            return x_paths, None

        y_paths = list(sorted(sub_path.glob(f'{y_dir}*/*')))
        y_paths = [str(path) for path in y_paths]
        y_ctr = len(y_paths)
        print(f"{type} output count: {y_ctr}")
        assert x_ctr == y_ctr

        return x_paths, y_paths

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

    def build_tfrecord(self, type, x_paths, y_paths=""):

        with tf.io.TFRecordWriter(f"{self.tfrecord_path}_{type}") as writer:
            if type == "test":
                for x_path in x_paths:
                    feature = {
                        'nframes': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.nframes])),
                    }
                    for i, _x in enumerate(x_path):
                        frame = open(_x, 'rb').read()
                        feature[f'frame{i}'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[frame]))
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            else:
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

    def build_clip_from_tensor(self, image_list, image_tensor): # deprecated
        rtn = []
        for i in range(len(image_list)):
            next_frames = image_tensor[i:i + self.center + 1]
            if i < self.center:
                prev_frames = image_tensor[:i]
            else:
                prev_frames = image_tensor[i - self.center:i]

            to_fill = self.nframes - next_frames.shape[0] - prev_frames.shape[0]
            if to_fill:
                if len(prev_frames) and i < self.nframes:
                    pad_x = prev_frames[0][np.newaxis, :]
                    pad_x = np.repeat(pad_x, to_fill, axis=0)
                    xx = np.concatenate((pad_x, prev_frames, next_frames))
                else:
                    if i > self.nframes:
                        pad_x = next_frames[-1][np.newaxis, :]
                        pad_x = np.repeat(pad_x, to_fill, axis=0)
                        xx = np.concatenate((prev_frames, next_frames, pad_x))
                    else:
                        pad_x = next_frames[0][np.newaxis, :]
                        pad_x = np.repeat(pad_x, to_fill, axis=0)
                        xx = np.concatenate((pad_x, prev_frames, next_frames))
            else:
                xx = np.concatenate((prev_frames, next_frames))
            rtn.append(xx)
        return np.stack(rtn)

    def decode_tfrecord(self, type):
        dataset = tf.data.TFRecordDataset(f"{self.tfrecord_path}_{type}")
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

        dataset = dataset.repeat().shuffle(buffer_size=self.buffer_size)
        dataset = dataset.map(_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset.prefetch(buffer_size=self.prefetch_buffer_size)

    def __call__(self):
        print("Start building dataloader...")
        # self.preprocess()
        self.train_dataset = self.decode_tfrecord('train')
        self.val_dataset = self.decode_tfrecord('val')
        self.test_dataset = self.decode_tfrecord('test')
        return self.train_dataset, self.val_dataset, self.test_dataset

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

    train_dataset, val_dataset, test_dataset = REDSDataLoader(config)()

    # for step, (inputs, targets) in enumerate(train_dataset):
    #     print(inputs.shape)
    #     print(targets.shape)
    #     with tf.GradientTape() as tape:
    #         y = model(inputs)
    #     print(y.shape)

    val_dataset_iter = iter(val_dataset)
    for i in range(100):
        inputs, targets = val_dataset_iter.get_next()
        print(inputs.shape)
        print(targets.shape)
        y = model(inputs)
        print(y.shape)

