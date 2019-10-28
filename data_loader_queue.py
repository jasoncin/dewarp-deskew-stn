from __future__ import absolute_import
from __future__ import division

import os
import errno
import cv2
import threading
import numpy as np
from queue import Queue
from random import shuffle
from scipy import misc

from util.path_util import read_image_list

class DataGenerator(object):
    """

    """

    def __init__(self, path_list_train, path_list_eval, n_classes, batch_size,
                 thread_num=4, queue_capacity=64, label_prefix='mask', data_kwargs={}):
        """
            Define data generator parameters / flags
            Args:
                path_list_train: Path to training file list
                path_list_val: Path to validation file list
                n_classes: Number of output classes
                thread_num: Number of concurrent threads to generate data
                queue_capacity: Maximum queue capacity (reduce this if you have out-of-memory issue)
                data_kwargs: Additional keyword arguments for data augmentation (geometric transforms, rotation, ...)
            Returns:
                a DataGenerator instance
        """
        self.n_classes = n_classes
        self.label_prefix = label_prefix

        self.list_training = None
        self.size_training = 0
        self.q_training = None

        self.list_validation = None
        self.size_validation = 0
        self.q_validation = None

        self.img_height = 600
        self.img_width = 600
        """ 
            Augmentation parameters 
        """

        # Batch size for training & validation
        self.batch_size_training = batch_size
        self.batch_size_validation = batch_size

        # Use random rotation
        self.dilated_num = data_kwargs.get('dilated_num', 1)

        self.scale_min = data_kwargs.get('scale_min', 1.0)
        self.scale_max = data_kwargs.get('scale_max', 1.0)
        self.scale_val = data_kwargs.get('scale_val', 1.0)

        # Ensure one-hot encoding consistency after geometric distortions

        self.dominating_channel = data_kwargs.get('dominating_channel', 0)
        self.dominating_channel = min(self.dominating_channel, n_classes - 1)

        # shuffle dataset after each epoch
        self.shuffle = data_kwargs.get('shuffle', True)

        self.thread_num = thread_num
        self.queue_capacity = queue_capacity
        self.stop_training = threading.Event()
        self.stop_validation = threading.Event()

        # Start data generator thread(s) to fill the training queue
        if path_list_train != None:
            self.list_training = read_image_list(path_list_train, prefix=None)
            self.size_training = len(self.list_training)
            self.q_training, self.thread_training = self._get_list_queue(self.list_training, self.thread_num,
                                                                         self.queue_capacity, self.stop_training,
                                                                         self.batch_size_training, self.scale_min,
                                                                         self.scale_max)

        # Start data generator thread(s) to fill the validation queue
        if path_list_eval != None:
            self.list_validation = read_image_list(path_list_eval, prefix=None)
            self.size_validation = len(self.list_validation)
            self.q_validation, self.thread_validation = self._get_list_queue(self.list_validation, self.thread_num,
                                                                             self.queue_capacity, self.stop_validation,
                                                                             self.batch_size_validation, self.scale_val,
                                                                             self.scale_val)

    def get_data(self, name='training'):
        batch_size = self.batch_size_training
        input_list = self.list_training
        index = 0
        max_height = 0
        max_width = 0
        if name is 'training':
            imgs = []
            new_imgs = []

            masks = []
            new_masks = []

            while len(imgs) < batch_size:
                # shuffle(input_list)
                image_path = input_list[index]
                mask_path = image_path.replace("images", "labels")
                print(image_path)
                print(mask_path)
                input_img = cv2.imread(image_path, 0)
                mask_img = cv2.imread(mask_path, 0)

                input_img = input_img / 255.0
                mask_img = mask_img / 255.0

                imgs.append(input_img)
                masks.append(mask_img)

                max_width = max(input_img.shape[1], max_width)
                max_width = max(mask_img.shape[1], max_width)
                max_height = max(input_img.shape[0], max_height)
                max_height = max(mask_img.shape[0], max_height)

            max_width = max(max_width, max_height)
            max_height = max(max_width, max_height)

            for img in imgs:
                height = img.shape[0]
                pad_h = max_height - height

                width = img.shape[1]
                pad_w = max_width - width

                if pad_h + pad_w > 0:
                    npad = ((0, pad_h), (0, pad_w))
                    img = np.pad(img, npad, mode='constant', constant_values=255)
                    img = self.resize_image(img, self.img_width / img.shape[1], self.img_height / img.shape[0])

                new_imgs.append(np.expand_dims(img, 2))

            for mask in masks:
                height = mask.shape[0]
                pad_h = max_height - height

                width = mask.shape[1]
                pad_w = max_width - width

                if pad_h + pad_w > 0:
                    npad = ((0, pad_h), (0, pad_w))
                    mask = np.pad(mask, npad, mode='constant', constant_values=255)
                    mask = self.resize_image(mask, self.img_width / mask.shape[1], self.img_height / mask.shape[0])

                new_masks.append(np.expand_dims(mask, 2))
            return new_imgs, new_masks

    def next_data(self, name):
        """
            Return next data from the queue
        """
        if name is 'validation':
            q = self.q_validation

        elif name is 'training':
            q = self.q_training

        if q is None:
            return None, None

        return q.get()

    def stop_all(self):
        """
            Stop all data generator threads
        """
        self.stop_training.set()
        self.stop_validation.set()

    def restart_val_runner(self):
        """
            Restart validation runner
        """
        if self.list_validation != None:
            self.stop_validation.set()
            self.stop_validation = threading.Event()
            self.q_validation, self.thread_validation = self._get_list_queue(self.list_validation, 1, 100,
                                                                             self.stop_validation,
                                                                             self.batch_size_validation, self.scale_val,
                                                                             self.scale_val)

    def _get_list_queue(self, input_list, thread_num, queue_capacity,
                        stop_event, batch_size, min_scale, max_scale):
        """
            Create a queue and add dedicated generator thread(s) to fill it
        """

        q = Queue(maxsize=queue_capacity)
        threads = []

        for t in range(thread_num):
            threads.append(threading.Thread(target=self._fillQueue, args=(
                q, input_list[:], stop_event, batch_size, min_scale, max_scale)))

        for t in threads:
            t.start()
        return q, threads

    def _get_mask_image(self, path, num_mask_per_sample=1):
        """
            Create a list of mask images with respect to each image sample
            :param path: the image path
            :param num_mask_per_sample: number of masks for that image sample (mask channel)
            :param prefix: name of folder containing mask images
            :return: a list of mask images
        """
        # print("Train path", path)
        if num_mask_per_sample < 1:
            raise ValueError('{} should not be less than 1!'.format(num_mask_per_sample))
        else:
            list_mask = []
            file_path, file_extension = os.path.splitext(path)
            dir_file, file_name = os.path.split(file_path)

            check_dir = os.path.join(os.path.dirname(dir_file), self.label_prefix)
            if not os.path.exists(check_dir):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), check_dir)

            if num_mask_per_sample == 1:
                mask_idx_path = os.path.join(os.path.dirname(dir_file), \
                                             self.label_prefix, '{}{}'.format(file_name, file_extension))
                # print("Mask name:", mask_idx_path)
                list_mask.append(misc.imread(mask_idx_path))
            else:
                for idx in range(0, num_mask_per_sample):
                    mask_idx_path = os.path.join(os.path.dirname(dir_file), \
                                                 self.label_prefix, '{}_{}{}'.format(file_name, idx, file_extension))
                    list_mask.append(misc.imread(mask_idx_path))

            return list_mask

    def resize_image(self, img, scale_width, scale_height):
        # scale_percent = 60  # percent of original size

        width = int(img.shape[1] * scale_width)
        height = int(img.shape[0] * scale_height)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        return resized

    def _fillQueue(self, q, input_list, stop_event, batch_size, min_scale, max_scale):
        """
            Main function to generate new input-output pair an put it into the queue
            Args:
                q: Output queue
                input_list: List of input JSON(s)
                stop_event: Thread stop event
                batch_size: Batch-size
                affine: Use affine transform
                elastic: Use elastic transform
                rotate: Use random rotation
                rotate_90: Use random rotation (constrained to multiple of 90 degree)
            Returns:
                None
        """

        if self.shuffle:
            shuffle(input_list)
        index = 0
        batch_pair = None

        max_height = 0
        max_width = 0
        while (not stop_event.is_set()):
            if batch_pair is None:
                imgs = []
                new_imgs = []

                masks = []
                new_masks = []

                while len(imgs) < batch_size:
                    if index == len(input_list):
                        if self.shuffle:
                            shuffle(input_list)
                        index = 0
                    try:
                        image_path = input_list[index]
                    except IndexError:
                        print(index, len(input_list))

                    mask_path = image_path.replace("images", "labels")
                    input_img = cv2.imread(image_path, 0)
                    mask_img = cv2.imread(mask_path, 0)

                    input_img = input_img / 255.0
                    mask_img = mask_img / 255.0

                    imgs.append(input_img)
                    masks.append(mask_img)

                    max_width = max(input_img.shape[1], max_width)
                    max_width = max(mask_img.shape[1], max_width)
                    max_height = max(input_img.shape[0], max_height)
                    max_height = max(mask_img.shape[0], max_height)

                max_width = max(max_width, max_height)
                max_height = max(max_width, max_height)

                for img in imgs:
                    height = img.shape[0]
                    pad_h = max_height - height

                    width = img.shape[1]
                    pad_w = max_width - width

                    if pad_h + pad_w > 0:
                        npad = ((0, pad_h), (0, pad_w))
                        img = np.pad(img, npad, mode='constant', constant_values=255)
                        img = self.resize_image(img, self.img_width / img.shape[1], self.img_height / img.shape[0])

                    new_imgs.append(np.expand_dims(img, 2))

                for mask in masks:
                    height = mask.shape[0]
                    pad_h = max_height - height

                    width = mask.shape[1]
                    pad_w = max_width - width

                    if pad_h + pad_w > 0:
                        npad = ((0, pad_h), (0, pad_w))
                        mask = np.pad(mask, npad, mode='constant', constant_values=255)
                        mask = self.resize_image(mask, self.img_width / mask.shape[1], self.img_height / mask.shape[0])

                    new_masks.append(np.expand_dims(mask, 2))

                # batch_x = np.concatenate(new_imgs)
                # batch_y = np.concatenate(new_masks)
                batch_pair = [new_imgs, new_masks]
                # print(batch_x.shape)

            try:
                q.put(batch_pair, timeout=1)
                batch_pair = None
                index += 1
            except:
                continue
