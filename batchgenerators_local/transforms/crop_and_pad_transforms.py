# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from batchgenerators_local.augmentations.crop_and_pad_augmentations import center_crop, pad_nd_image_and_seg, random_crop
from batchgenerators_local.transforms.abstract_transforms import AbstractTransform
import numpy as np


class CenterCropTransform(AbstractTransform):
    """ Crops data and seg (if available) in the center

    Args:
        output_size (int or tuple of int): Output patch size

    """

    def __init__(self, crop_size, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.crop_size = crop_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        data, seg = center_crop(data, self.crop_size, seg)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict


class CenterCropSegTransform(AbstractTransform):
    """ Crops seg in the center (required if you are using unpadded convolutions in a segmentation network).
    Leaves data as it is

    Args:
        output_size (int or tuple of int): Output patch size

    """

    def __init__(self, output_size, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.output_size = output_size

    def __call__(self, **data_dict):
        seg = data_dict.get(self.label_key)

        if seg is not None:
            data_dict[self.label_key] = center_crop(seg, self.output_size, None)[0]
        else:
            from warnings import warn
            warn("You shall not pass data_dict without seg: Used CenterCropSegTransform, but there is no seg", Warning)
        return data_dict


class RandomCropTransform(AbstractTransform):
    """ Randomly crops data and seg (if available)

    Args:
        crop_size (int or tuple of int): Output patch size

        margins (tuple of int): how much distance should the patch border have to the image broder (bilaterally)?

    """

    def __init__(self, crop_size=128, margins=(0, 0, 0), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.margins = margins
        self.crop_size = crop_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        data, seg = random_crop(data, seg, self.crop_size, self.margins)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict


class PadTransform(AbstractTransform):
    def __init__(self, new_size, pad_mode_data='constant', pad_mode_seg='constant',
                 np_pad_kwargs_data=None, np_pad_kwargs_seg=None,
                 data_key="data", label_key="seg"):
        """
        Pads data and seg to new_size. Only supports numpy arrays for data and seg.

        :param new_size: (x, y(, z))
        :param pad_value_data:
        :param pad_value_seg:
        :param data_key:
        :param label_key:
        """
        self.data_key = data_key
        self.label_key = label_key
        self.new_size = new_size
        self.pad_mode_data = pad_mode_data
        self.pad_mode_seg = pad_mode_seg
        if np_pad_kwargs_data is None:
            np_pad_kwargs_data = {}
        if np_pad_kwargs_seg is None:
            np_pad_kwargs_seg = {}
        self.np_pad_kwargs_data = np_pad_kwargs_data
        self.np_pad_kwargs_seg = np_pad_kwargs_seg

        assert isinstance(self.new_size, (tuple, list, np.ndarray)), "new_size must be tuple, list or np.ndarray"

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        assert len(self.new_size) + 2 == len(data.shape), "new size must be a tuple/list/np.ndarray with shape " \
                                                    "(x, y(, z))"
        data, seg = pad_nd_image_and_seg(data, seg, self.new_size, None,
                                         np_pad_kwargs_data=self.np_pad_kwargs_data,
                                         np_pad_kwargs_seg=self.np_pad_kwargs_seg,
                                         pad_mode_data=self.pad_mode_data,
                                         pad_mode_seg=self.pad_mode_seg)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict

