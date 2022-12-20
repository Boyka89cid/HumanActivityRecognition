import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import torch
import Augmentor
from Augmentor import Operations

'''
Below Class has been referred from
https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py
'''


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Translate(object):
    def __init__(self, probability=0.5, translate=(0.2, 0.2)):
        self.affline = torchvision.transforms.RandomAffine(degrees=0, translate=translate,
                                                           scale=None, shear=None, resample=False, fillcolor=0)
        self.probability = probability

    def __call__(self, img_group):
        p = random.random()
        if p < self.probability:
            return [self.affline(img) for img in img_group]
        else:
            return img_group


class Brightness(object):
    def __init__(self, probability=0.5, brightness=0.5):
        self.color_jitter = torchvision.transforms.ColorJitter(brightness=brightness)
        self.probability = probability

    def __call__(self, img_group):
        p = random.random()
        if p < self.probability:
            return [self.color_jitter(img) for img in img_group]
        else:
            return img_group


class Contrast(object):
    def __init__(self, probability=0.5, contrast=0.5):
        self.color_jitter = torchvision.transforms.ColorJitter(contrast=contrast)
        self.probability = probability

    def __call__(self, img_group):
        p = random.random()
        if p < self.probability:
            return [self.color_jitter(img) for img in img_group]
        else:
            return img_group


class Saturation(object):
    def __init__(self, probability=0.25, saturation=0.25):
        self.color_jitter = torchvision.transforms.ColorJitter(saturation=saturation)
        self.probability = probability

    def __call__(self, img_group):
        p = random.random()
        if p < self.probability:
            return [self.color_jitter(img) for img in img_group]
        else:
            return img_group


class Hue(object):
    def __init__(self, probability=0.25, hue=0.25):
        self.color_jitter = torchvision.transforms.ColorJitter(hue=hue)
        self.probability = probability

    def __call__(self, img_group):
        p = random.random()
        if p < self.probability:
            return [self.color_jitter(img) for img in img_group]
        else:
            return img_group


class ColorJitter(object):
    def __init__(self):
        self.coljit = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.75, saturation=0, hue=0.25)

    def __call__(self, img_group):
        return [self.coljit(img) for img in img_group]


class RandomCrop(object):
    def __init__(self):
        self.p = Augmentor.Operations.CropRandom(probability=0.5, percentage_area=0.5)

    def __call__(self, img_group):
        return self.p.perform_operation(img_group)


class Scale(object):
    def __init__(self):
        self.p = Augmentor.Operations.Scale(probability=0.5, scale_factor=1.2)

    def __call__(self, img_group):
        return self.p.perform_operation(img_group)


class Rotate(object):
    def __init__(self, probability=0.5, max_left_rotation=15, max_right_rotation=15):
        self.p = Augmentor.Operations.RotateRange(probability=probability,
                                                  max_left_rotation=max_left_rotation,
                                                  max_right_rotation=max_right_rotation)

    def __call__(self, img_group):
        return self.p.perform_operation(img_group)


class Skew(object):
    def __init__(self, probability=0.5, skew_type='TILT', magnitude=1.0):
        self.p = Augmentor.Operations.Skew(probability=probability, skew_type=skew_type, magnitude=magnitude)

    def __call__(self, img_group):
        return self.p.perform_operation(img_group)


class Shear(object):
    def __init__(self, probability=0.25, max_shear_left=8, max_shear_right=8):
        self.p = Augmentor.Operations.Shear(probability=probability, max_shear_left=max_shear_left,
                                            max_shear_right=max_shear_right)

    def __call__(self, img_group):
        return self.p.perform_operation(img_group)


class GrayScale(object):
    def __init__(self, probability=0.05):
        self.p = Augmentor.Operations.Greyscale(probability=probability)

    def __call__(self, img_group):
        return self.p.perform_operation(img_group)


class RandomNoise(object):
    def __init__(self):
        self.p = Augmentor.Operations.RandomErasing(probability=0.2, rectangle_area=0.5)

    def __call__(self, img_group):
        return self.p.perform_operation(img_group)


class Noise(object):
    def __init__(self, probability=0.05):
        self.p = Augmentor.Operations.RandomErasing(probability=1, rectangle_area=0.1)
        self.p1 = Augmentor.Operations.GaussianDistortion(probability=1, grid_width=3, grid_height=3, magnitude=5,
                                                          corner='bell', method='in', mex=0.5, mey=0.5, sdx=0.05,
                                                          sdy=0.05)
        self.probability = probability

    def __call__(self, img_group):
        prob = random.random()
        prob2 = random.random()

        if prob < self.probability:
            if prob2 > 0.5:
                return self.p.perform_operation(img_group)
            else:
                return self.p1.perform_operation(img_group)
        else:
            return img_group


class GaussianNoise(object):
    def __init__(self):
        self.p = Augmentor.Operations.GaussianDistortion(probability=1, grid_width=3, grid_height=3, magnitude=5,
                                                         corner='bell', method='in', mex=0.5, mey=0.5, sdx=0.05,
                                                         sdy=0.05)

    def __call__(self, img_group):
        return self.p.perform_operation(img_group)


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalizeNumpy(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        rep_mean = self.mean * (input.shape[-1] // len(self.mean))
        rep_std = self.std * (input.shape[-1] // len(self.std))
        rep_mean = np.array(rep_mean, dtype=np.float32)
        rep_std = np.array(rep_std, dtype=np.float32)

        normalized = ((input / 255) - rep_mean) / rep_std

        # print(input.max(), input.min(), rep_mean.shape)
        return normalized


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


'''
Below Class has been referred from
https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py
'''


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):
    def __call__(self, data):
        return data
