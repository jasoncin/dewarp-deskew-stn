from __future__ import absolute_import
from __future__ import division

import numpy as np
from random import shuffle 

from util.path_util import read_image_list
import cv2


class DataGenerator(object):
    """

    """
    def __init__(self, path_list_train, path_list_eval, n_classes):
        self.n_classes = 1
        if path_list_train != None:
            self.list_training = read_image_list(path_list_train, prefix=None)
            self.size_training = len(self.list_training)

        if path_list_eval != None:
            self.list_test = read_image_list(path_list_eval, prefix=None)
            self.size_validation = len(self.list_test)

    def get_data(self):
        return (self._get_list(self.list_training), self._get_list(self.list_test))

    def _get_list(self, input_list):

        shuffle(input_list)
        imgs = []
        new_imgs = []

        masks = []
        new_masks = []

        max_height = 0
        max_width = 0

        for index in range(len(input_list)):
            # if index % 100 == 0:
            #     print("{}".format(index))

            image_path = input_list[index]
            mask_path = image_path.replace("images","labels")
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

        for img in imgs:
            height = img.shape[0]
            pad_h = max_height - height

            width = img.shape[1]
            pad_w = max_width - width

            if pad_h + pad_w > 0:
                npad = ((0, pad_h), (0, pad_w))
                img = np.pad(img, npad, mode='constant', constant_values=255)
            new_imgs.append(np.expand_dims(img, 2))

        for mask in masks:
            height = mask.shape[0]
            pad_h = max_height - height

            width = mask.shape[1]
            pad_w = max_width - width

            if pad_h + pad_w > 0:
                npad = ((0, pad_h), (0, pad_w))
                mask = np.pad(mask, npad, mode='constant', constant_values=255)
            new_masks.append(np.expand_dims(mask, 2))


        return new_imgs, new_masks, max_height, max_width
