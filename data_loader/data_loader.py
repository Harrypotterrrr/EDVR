import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os
import PIL
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetGenerator:
    def __init__(self, config):
        # data, info = tfds.load(name="mnist", download=True, with_info=True)
        # self.train_data = data['train']
        # self.test_data = data['test']

        # self.config = config
        # assert isinstance(self.train_data, tf.data.Dataset)

        self.nframes = config["nframes"]
        self.center = self.nframes // 2
        self.batch_size = config["batch_size"]

        root_path = pathlib.Path(config["root_path"])
        train_path = root_path.joinpath('train')


        train_blur_bicubic_paths = []
        train_blur_bicubic_subpath = list(sorted(train_path.glob(f'{config["train_blur_bicubic"]}*')))
        for subpath in train_blur_bicubic_subpath:
            label = pathlib.Path(subpath).name
            train_blur_bicubic_subpath = [str(path) for path in train_path.glob(f'{config["train_blur_bicubic"]}{label}/*')]
            train_blur_bicubic_subpath = self.build_clips(sorted(train_blur_bicubic_subpath))
            # print(train_blur_bicubic_subpath)
            train_blur_bicubic_paths.append(train_blur_bicubic_subpath)

        train_blur_bicubic_paths = np.asarray(train_blur_bicubic_paths).reshape(-1, self.nframes)

        train_blur_paths = list(sorted(train_path.glob(f'{config["train_blur_bicubic"]}*/*')))
        train_blur_paths = [str(path) for path in train_blur_paths]

        train_blur_bicubic_ctr = len(train_blur_bicubic_paths)
        train_blur_ctr = len(train_blur_paths)

        # print(train_blur_ctr, train_blur_bicubic_ctr)
        assert train_blur_ctr == train_blur_bicubic_ctr
        train_blur_bicubic_paths = train_blur_bicubic_paths[:10] # TODO
        train_blur_paths = train_blur_paths[:10]

        # print(len(train_blur_bicubic_paths), len(train_blur_paths))
        ds = tf.data.Dataset.from_tensor_slices((train_blur_bicubic_paths, train_blur_paths))

        def convert_types(image):
            image = tf.cast(image, tf.float32)
            image /= 255
            return image

        def load_image(x_path, y_path):
            # assert len(x_path) == self.nframes

            # x_image = []
            # for i in x_path:
            #     x = tf.io.read_file(i)
            #     x_image.append(tf.image.decode_image(x))
            # x_image = tf.convert_to_tensor(x_image, dtype=tf.float32)

            x_image1 = convert_types(tf.image.decode_image(tf.io.read_file(x_path[0]), channels=3))
            x_image2 = convert_types(tf.image.decode_image(tf.io.read_file(x_path[1]), channels=3))
            x_image3 = convert_types(tf.image.decode_image(tf.io.read_file(x_path[2]), channels=3))
            x_image4 = convert_types(tf.image.decode_image(tf.io.read_file(x_path[3]), channels=3))
            x_image5 = convert_types(tf.image.decode_image(tf.io.read_file(x_path[4]), channels=3)) # disgusting decorator @tf.funciton TODO
            x_image = tf.stack([x_image1, x_image2, x_image3, x_image4, x_image5])

            y = tf.io.read_file(y_path)
            y_image = tf.image.decode_image(y, channels=3)

            # image = tf.image.resize(image, [192, 192])
            # image /= 255.0
            return x_image, y_image

        # for step, (inputs, targets) in enumerate(ds):
        #     load_and_preprocess_from_path_label(inputs, targets)

        ds = ds.map(load_image)
        self.ds = ds.shuffle(len(ds)).batch(self.batch_size) #.repeat(10)

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
        return self.ds

    def preprocess(self):
        self.train_data = self.train_data.map(
            DatasetGenerator.convert_types
        ).batch(self.config.batch_size)
        self.test_data = self.test_data.map(
            DatasetGenerator.convert_types
        ).batch(self.config.batch_size)

    @staticmethod
    def convert_types(batch):
        image, label = batch.values()
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

if __name__ == "__main__":
    dataset = DatasetGenerator(None)()


    from models.EDVR import EDVR
    model = EDVR()
    for step, (inputs, targets) in enumerate(dataset):
        y = model(inputs)


