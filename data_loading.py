import copy
import glob
import os

import cv2
import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, batch_size, input_shape, num_classes, start_number=1, end_number=31, shuffle=True):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.start_number = start_number
        self.end_number = end_number
        self.emotions_map = {
            'AF': 0,
            'AN': 1,
            'DI': 2,
            'HA': 3,
            'NE': 4,
            'SA': 5,
            'SU': 6

        }
        # load the data from the root directory
        self.class_names = []
        self.data, self.labels = self.get_data(db_dir)
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def get_data(self, root_dir):
        """
        Loads the paths to the images and their corresponding labels from the database directory
        """
        paths = []
        for index in range(self.start_number, self.end_number):
            number = str(index)
            if index < 10:
                number = "0" + number
            paths += glob.glob(root_dir + "/AF" + number + "/*") \
                     + glob.glob(root_dir + "/AM" + number + "/*") \
                     + glob.glob(root_dir + "/BF" + number + "/*") \
                     + glob.glob(root_dir + "/BM" + number + "/*")
        labels = []
        copy_paths = copy.deepcopy(paths)
        for path in copy_paths:
            key = os.path.basename(path)[4:6]
            if key in self.emotions_map:
                labels.append(self.emotions_map[key])
            else:
                paths.remove(path)

        self.class_names = list(set(labels))
        sorted(self.class_names)

        self.data = np.asarray(paths)
        self.labels = np.asarray([self.class_names.index(label) for label in labels])
        return self.data, self.labels

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = []
        for i in batch_indices:
            image = cv2.imread(self.data[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.square_image(image)
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
            batch_x.append(image)
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(self.labels[batch_indices])
        return batch_x, batch_y

    def square_image(self, image):
        width_pad = 0
        height_pad = 0
        if image.shape[0] > image.shape[1]:
            width_pad = (image.shape[0] - image.shape[1]) // 2
        else:
            height_pad = (image.shape[1] - image.shape[0]) // 2
        return np.pad(image, ((height_pad, height_pad), (width_pad, width_pad), (0, 0)), mode="edge")

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
