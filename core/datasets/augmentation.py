import ipdb
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as tvf
import random
from math import ceil
import torch
import cv2
from core.tools import transforms_tools as F

# some are implemented by r2d2: https://github.com/naver/r2d2/blob/master/tools/transforms.py
class Scale (object):
    """ Rescale the input PIL.Image to a given size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    The smallest dimension of the resulting image will be = size.

    if largest == True: same behaviour for the largest dimension.

    if not can_upscale: don't upscale
    if not can_downscale: don't downscale
    """

    def __init__(self, size, interpolation=Image.BILINEAR, largest=False,
                 can_upscale=True, can_downscale=True):
        assert isinstance(size, int) or (len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.largest = largest
        self.can_upscale = can_upscale
        self.can_downscale = can_downscale

    def __repr__(self):
        fmt_str = "RandomScale(%s" % str(self.size)
        if self.largest:
            fmt_str += ', largest=True'
        if not self.can_upscale:
            fmt_str += ', can_upscale=False'
        if not self.can_downscale:
            fmt_str += ', can_downscale=False'
        return fmt_str+')'

    def get_params(self, imsize):
        w, h = imsize
        if isinstance(self.size, int):
            def cmp(a, b): return (a >= b) if self.largest else (a <= b)
            if (cmp(w, h) and w == self.size) or (cmp(h, w) and h == self.size):
                ow, oh = w, h
            elif cmp(w, h):
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
        else:
            ow, oh = self.size
        return ow, oh

    def __call__(self, inp, mask=None):

        img = F.grab_img(inp)
        w, h = img.size

        size2 = ow, oh = self.get_params(img.size)

        if size2 != img.size:
            a1, a2 = img.size, size2
            if (self.can_upscale and min(a1) < min(a2)) or (self.can_downscale and min(a1) > min(a2)):
                img = img.resize(size2, self.interpolation)
                if mask is not None:
                    mask = mask.resize(size2, Image.NEAREST)

        return F.update_img_and_labels(inp, img, persp=(ow/w, 0, 0, 0, oh/h, 0, 0, 0))


class RandomScale (Scale):
    """Rescale the input PIL.Image to a random size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        min_size (int): min size of the smaller edge of the picture.
        max_size (int): max size of the smaller edge of the picture.

        ar (float or tuple):
            max change of aspect ratio (width/height).

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, min_size, max_size, ar=1,
                 can_upscale=False, can_downscale=True, interpolation=Image.BILINEAR):
        Scale.__init__(self, 0, can_upscale=can_upscale,
                       can_downscale=can_downscale, interpolation=interpolation)
        assert type(min_size) == type(
            max_size), 'min_size and max_size can only be 2 ints or 2 floats'
        assert isinstance(min_size, int) and min_size >= 1 or isinstance(
            min_size, float) and min_size > 0
        assert isinstance(max_size, (int, float)) and min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        if type(ar) in (float, int):
            ar = (min(1/ar, ar), max(1/ar, ar))
        assert 0.2 < ar[0] <= ar[1] < 5
        self.ar = ar

    def get_params(self, imsize):
        w, h = imsize
        if isinstance(self.min_size, float):
            min_size = int(self.min_size*min(w, h) + 0.5)
        if isinstance(self.max_size, float):
            max_size = int(self.max_size*min(w, h) + 0.5)
        if isinstance(self.min_size, int):
            min_size = self.min_size
        if isinstance(self.max_size, int):
            max_size = self.max_size

        if not self.can_upscale:
            max_size = min(max_size, min(w, h))

        size = int(0.5 + F.rand_log_uniform(min_size, max_size))
        ar = F.rand_log_uniform(*self.ar)  # change of aspect ratio

        if w < h:  # image is taller
            ow = size
            oh = int(0.5 + size * h / w / ar)
            if oh < min_size:
                ow, oh = int(0.5 + ow*float(min_size)/oh), min_size
        else:  # image is wider
            oh = size
            ow = int(0.5 + size * w / h * ar)
            if ow < min_size:
                ow, oh = min_size, int(0.5 + oh*float(min_size)/ow)

        assert ow >= min_size, 'image too small (width=%d < min_size=%d)' % (
            ow, min_size)
        assert oh >= min_size, 'image too small (height=%d < min_size=%d)' % (
            oh, min_size)
        return ow, oh


class RandomCrop (object):
    """Crop the given PIL Image at a random location.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __repr__(self):
        return "RandomCrop(%s)" % str(self.size)

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        assert h >= th and w >= tw, "Image of %dx%d is too small for crop %dx%d" % (
            w, h, tw, th)

        y = np.random.randint(0, h - th) if h > th else 0
        x = np.random.randint(0, w - tw) if w > tw else 0
        return x, y, tw, th

    def __call__(self, inp):
        img = F.grab_img(inp)

        padl = padt = 0
        if self.padding:
            if F.is_pil_image(img):
                img = ImageOps.expand(img, border=self.padding, fill=0)
            else:
                assert isinstance(img, F.DummyImg)
                img = img.expand(border=self.padding)
            if isinstance(self.padding, int):
                padl = padt = self.padding
            else:
                padl, padt = self.padding[0:2]

        i, j, tw, th = self.get_params(img, self.size)
        img = img.crop((i, j, i+tw, j+th))

        return F.update_img_and_labels(inp, img, persp=(1, 0, padl-i, 0, 1, padt-j, 0, 0))


class CenterCrop (RandomCrop):
    """Crops the given PIL Image at the center.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        y = int(0.5 + ((h - th) / 2.))
        x = int(0.5 + ((w - tw) / 2.))
        return x, y, tw, th


class RandomRotation(object):
    """Rescale the input PIL.Image to a random size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        degrees (float):
            rotation angle.

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, degrees, interpolation=Image.NEAREST):
        self.degrees = degrees
        self.interpolation = interpolation

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size

        angle = np.random.uniform(-self.degrees, self.degrees)
        #angle = 90

        img = img.rotate(angle, resample=self.interpolation)
        w2, h2 = img.size

        trf = F.translate(w/2, h/2)
        trf = F.persp_mul(trf, F.rotate(-angle * np.pi/180))
        trf = F.persp_mul(trf, F.translate(-w2/2, -h2/2))
        # trf = np.array(trf) * np.array([1,1,-1,1,1,-1,1,1])
        # trf = tuple(trf)
        return F.update_img_and_labels(inp, img, persp=trf, theta=angle)


class RandomFlip(object):
    """Rescale the input PIL.Image to a random size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        degrees (float):
            rotation angle.

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, interpolation=Image.BILINEAR):
        self.interpolation = interpolation

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, _ = img.size

        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            trf = (-1, 0, w-1, 0, 1, 0, 0, 0)
        else:
            trf = (1, 0, 0, 0, 1, 0, 0, 0)

        return F.update_img_and_labels(inp, img, persp=trf)


class RandomTilting(object):  # https://en.wikipedia.org/wiki/Tilt%E2%80%93shift_photography
    """Apply a random tilting (left, right, up, down) to the input PIL.Image
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        maginitude (float):
            maximum magnitude of the random skew (value between 0 and 1)
        directions (string):
            tilting directions allowed (all, left, right, up, down)
            examples: "all", "left,right", "up-down-right"
    """

    def __init__(self, magnitude, directions='all'):
        self.magnitude = magnitude
        self.directions = directions.lower().replace(',', ' ').replace('-', ' ')

    def __repr__(self):
        return "RandomTilt(%g, '%s')" % (self.magnitude, self.directions)

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size

        x1, y1, x2, y2 = 0, 0, h, w
        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.directions == 'all':
            choices = [0, 1, 2, 3]
        else:
            dirs = ['left', 'right', 'up', 'down']
            choices = []
            for d in self.directions.split():
                try:
                    choices.append(dirs.index(d))
                except:
                    raise ValueError('Tilting direction %s not recognized' % d)

        skew_direction = random.choice(choices)

        # print('randomtitlting: ', skew_amount, skew_direction) # to debug random

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1),                # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -
                          p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -
                          p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(original_plane).reshape(8)

        homography2 = np.dot(np.linalg.pinv(A), B)
        homography2 = tuple(np.array(homography2).reshape(8))
        # print(homography)

        img = img.transform(img.size, Image.PERSPECTIVE,
                            homography2, resample=Image.BILINEAR)

        homography1 = np.linalg.pinv(np.float32(
            homography2+(1,)).reshape(3, 3)).ravel()[:8]
        return F.update_img_and_labels(inp, img, persp=tuple(homography1), persp2=homography2)


RandomTilt = RandomTilting  # redefinition


class Tilt(object):
    """Apply a known tilting to an image
    """

    def __init__(self, *homography):
        assert len(homography) == 8
        self.homography = homography

    def __call__(self, inp):
        img = F.grab_img(inp)
        homography = self.homography
        # print(homography)

        img = img.transform(img.size, Image.PERSPECTIVE,
                            homography, resample=Image.BICUBIC)

        homography = np.linalg.pinv(np.float32(
            homography+(1,)).reshape(3, 3)).ravel()[:8]
        return F.update_img_and_labels(inp, img, persp=tuple(homography))


class StillTransform (object):
    """ Takes and return an image, without changing its shape or geometry.
    """

    def _transform(self, img):
        raise NotImplementedError()

    def __call__(self, inp):
        img = F.grab_img(inp)

        # transform the image (size should not change)
        try:
            img = self._transform(img)
        except TypeError:
            pass

        return F.update_img_and_labels(inp, img, persp=(1, 0, 0, 0, 1, 0, 0, 0))


class PixelNoise (StillTransform):
    """ Takes an image, and add random white noise.
    """

    def __init__(self, ampl=20):
        StillTransform.__init__(self)
        assert 0 <= ampl < 255
        self.ampl = ampl

    def __repr__(self):
        return "PixelNoise(%g)" % self.ampl

    def _transform(self, img):
        img = np.float32(img)
        img += np.random.uniform(0.5-self.ampl/2, 0.5 +
                                 self.ampl/2, size=img.shape)
        return Image.fromarray(np.uint8(img.clip(0, 255)))


class ColorJitter (StillTransform):
    """Randomly change the brightness, contrast and saturation of an image.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self):
        return "ColorJitter(%g,%g,%g,%g)" % (
            self.brightness, self.contrast, self.saturation, self.hue)

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
        Transform which randomly adjusts brightness, contrast and
        saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(
                max(0, 1 - brightness), 1 + brightness)
            transforms.append(tvf.Lambda(
                lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(
                max(0, 1 - contrast), 1 + contrast)
            transforms.append(tvf.Lambda(
                lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(
                max(0, 1 - saturation), 1 + saturation)
            transforms.append(tvf.Lambda(
                lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(tvf.Lambda(
                lambda img: F.adjust_hue(img, hue_factor)))

        # print('colorjitter: ', brightness_factor, contrast_factor, saturation_factor, hue_factor) # to debug random seed

        np.random.shuffle(transforms)
        transform = tvf.Compose(transforms)

        return transform

    def _transform(self, img):
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img)


class RandomGrayscale(StillTransform):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

    def _transform(self, img):
        num_output_channels = tvf.functional._get_image_num_channels(img)
        if torch.rand(1) < self.p:
            return tvf.functional.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        return img


class GaussianNoise(StillTransform):
    def __init__(self, is_img=True, rgb_noise=3, depth_noise=1000, p=0.5):
        super().__init__()
        self.rgb_noise = rgb_noise
        self.depth_noise = depth_noise
        self.prob = p
        self.is_img = is_img

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.prob)

    def _transform(self, img):
        if np.random.uniform() < self.prob:
            img = np.float32(img)
            if self.is_img:
                std = np.random.uniform(0, self.rgb_noise)
                noise = np.random.normal(0, std, size=img.shape)
                img += noise
                img = Image.fromarray(np.uint8(img.clip(0, 255)))
            else:
                std = np.random.uniform(0, self.depth_noise)
                noise = np.random.normal(0, std, size=img.shape)
                img += noise
                img = Image.fromarray(np.uint16(img.clip(0, 65535)))

        return img


class GaussianBlur(StillTransform):
    def __init__(self, is_img=True, max_kernel_size=3, p=0.4):
        super().__init__()
        self.max_kernel_size = max_kernel_size
        self.prob = p
        self.is_img = is_img

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.prob)

    def _transform(self, img):
        if np.random.uniform() < self.prob:
            ksize = np.random.randint(2, self.max_kernel_size+1)
            ksize = 2*ksize-1
            if self.is_img:
                img = np.uint8(img)
            else:
                img = np.uint16(img)

            img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=2)
            img = Image.fromarray(img)

        return img


class DepthMissing(StillTransform):
    def __init__(self, miss=0.4, p=0.3):
        super().__init__()
        self.missing_percent = miss
        self.prob = p

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.prob)

    def _transform(self, depth):
        #W = depth.shape[1]
        #H = depth.shape[0]
        if np.random.uniform(0, 1) < self.prob:
            depth = np.float32(depth)
            vs, us = np.where(depth > 10000)
            missing_percent = np.random.uniform(0, self.missing_percent)
            missing_ids1 = np.random.choice(np.arange(0, len(us)), int(
                missing_percent*len(us)), replace=False)
            missing_ids2 = np.random.choice(np.arange(0, len(vs)), int(
                missing_percent*len(vs)), replace=False)
            depth[vs[missing_ids1], us[missing_ids2]] = 0
            depth = Image.fromarray(np.uint16(depth))

        return depth


class ObjectAugmentation(object):
    def __init__(self, scale_size, crop_size, rotation, p_gray, p_gaussian_noise,
                 p_gaussian_blur, color_jitter, trials_randomcrop, depth_normalize, p_mask_ref,
                 idx_as_rng_seed, rgb_norm_mean, rgb_norm_std, depth_norm_mean, depth_norm_std,
                 mask_ref_img=False, flip=False, use_depth=False, use_render=False):

        self.crop = RandomCrop(crop_size)
        self.scale = RandomScale(scale_size[0], scale_size[1])
        self.rotation = RandomRotation(rotation)

        if flip:
            self.randomflip = RandomFlip()

        self.gray = RandomGrayscale(p_gray)
        self.gaussian_noise = GaussianNoise(p=p_gaussian_noise)
        self.gaussian_blur = GaussianBlur(p=p_gaussian_blur)
        self.color_transform = ColorJitter(color_jitter[0],
                                           color_jitter[1],
                                           color_jitter[2],
                                           color_jitter[3])

        # depth
        self.use_depth = use_depth
        self.depth_normalize = depth_normalize

        self.rgb_transform = tvf.Compose([
            self.color_transform,
            self.gray,
            self.gaussian_noise,
            self.gaussian_blur,
            self.rotation,
        ])
        self.rgb_norm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=rgb_norm_mean, std=rgb_norm_std)])

        self.depth_transform = tvf.Compose([
            self.gaussian_noise,
            self.gaussian_blur,
        ])
        self.depth_norm = tvf.Normalize(
            mean=depth_norm_mean, std=depth_norm_std)

        self.n_samples = trials_randomcrop
        self.mask_ref_img = mask_ref_img  # bool
        self.p_mask_ref = p_mask_ref

        self.use_render = use_render

    def __call__(self, metadata):

        img_1, img_2 = metadata['img1'], metadata['img2']
        aflow, label1, label2 = metadata['aflow'], np.array(
            metadata['label1']), metadata['label2']

        valid_mask1 = np.where(label1 > 0, True, False)

        #! scale
        img_2 = {'img': img_2, 'persp': (1, 0, 0, 0, 1, 0, 0, 0)}
        img_2 = self.scale(img_2)
        label2 = label2.resize(img_2['img'].size, Image.NEAREST)

        if self.use_depth:
            depth1 = metadata['depth1']
            depth2 = metadata['depth2']
            depth2 = depth2.resize(img_2['img'].size, Image.NEAREST)
        else:
            depth1, depth2 = None, None

        if self.use_render:
            self.render_img_num = metadata['render2']['color'].shape[0]
            render1 = metadata['render1']
            render2 = metadata['render2']
            render_color_2 = []
            render_label_2 = []
            render_depth_2 = []
            for cc in range(self.render_img_num):
                render_color_2.append(Image.fromarray(
                    render2['color'][cc]).resize(img_2['img'].size, Image.NEAREST))
                render_label_2.append(Image.fromarray(
                    render2['label'][cc]).resize(img_2['img'].size, Image.NEAREST))
                if self.use_depth:
                    render_depth_2.append(Image.fromarray(
                        render2['depth'][cc]).resize(img_2['img'].size, Image.NEAREST))
        else:
            render1, render2 = None, None

        #! rgb_transform
        img_2 = self.rgb_transform(img_2)
        label2 = np.array(label2.rotate(
            img_2['theta'], resample=Image.NEAREST))
        if self.use_depth:
            depth2 = depth2.rotate(img_2['theta'], resample=Image.NEAREST)
        if self.use_render:
            for cc in range(self.render_img_num):
                render_color_2[cc] = render_color_2[cc].rotate(
                    img_2['theta'], resample=Image.NEAREST)
                render_label_2[cc] = render_label_2[cc].rotate(
                    img_2['theta'], resample=Image.NEAREST)
                if self.use_depth:
                    render_depth_2[cc] = render_depth_2[cc].rotate(
                        img_2['theta'], resample=Image.NEAREST)

        #! depth_transform
        if self.use_depth:
            depth2 = self.depth_transform(depth2)
            if self.use_render:
                for cc in range(self.render_img_num):
                    render_depth_2[cc] = self.depth_transform(
                        render_depth_2[cc])

        # apply the same transformation to the flow
        # item in aflow is coordinate of img_2
        aflow[:] = F.persp_apply(
            img_2['persp'], aflow.reshape(-1, 2)).reshape(aflow.shape)
        #viz.show_flow_mask_image(np.array(img_1), np.array(img_2['img']), label1, np.array(label2), aflow.transpose(2,0,1))

        img_2 = img_2['img']
        if self.use_render:
            render2['color'] = render_color_2
            render2['label'] = render_label_2
            if self.use_depth:
                render2['label'] = render_depth_2

        #! random crop size
        img_1, img_2, label1, label2, aflow, valid_mask1, depth1, depth2, render1, render2 = \
            self.crop_img_max_info(img_1, img_2, label1, label2, aflow, valid_mask1,
                                   depth1, depth2, render1, render2, metadata['idx'])
        # viz.show_flow_mask_image(np.array(img_1), np.array(img_2), label1, label2, aflow)
        # ipdb.set_trace()

        #! mask img_1
        if self.mask_ref_img:
            mask_random = np.random.random()
            if mask_random < self.p_mask_ref:
                img_1 = img_1 * np.expand_dims(valid_mask1, 2)

        # viz.show_mask_image(img_1, img_2, label1, label2)
        # ipdb.set_trace()
        # for cc in range(render1['color'].shape[0]):
        #     viz.show_mask_image(img_1, np.uint8(render1['color'][cc]),
        #                           label1, np.uint8(render1['label'][cc]))
        # show_depth_image(img_1, img_2, depth1, depth2, save=False)

        if self.use_depth:
            if self.mask_ref_img and mask_random < self.p_mask_ref:
                depth1 *= valid_mask1

            depth1 = tvf.ToTensor()(depth1.astype(np.float32))
            depth2 = tvf.ToTensor()(depth2.astype(np.float32))

            if self.depth_normalize == 'variable':
                depth1 = (depth1 - depth1.min())/(depth1.max() - depth1.min())
                depth2 = (depth2 - depth2.min())/(depth2.max() - depth2.min())
            else:
                depth1 = depth1 / 65535 * 2.0
                depth2 = depth2 / 65535 * 2.0
            depth1 = self.depth_norm(depth1)
            depth2 = self.depth_norm(depth2)
            # ipdb.set_trace()
            # show_depth_image(img_1, img_2, depth1[0], depth2[0], save=False)

            inputs = dict(img1=torch.cat((self.rgb_norm(img_1), depth1.repeat(3, 1, 1)), 0),
                          img2=torch.cat((self.rgb_norm(img_2), depth2.repeat(3, 1, 1)), 0))
        else:
            inputs = dict(img1=self.rgb_norm(img_1), img2=self.rgb_norm(img_2))

        if self.use_render:
            self.render_to_input(label1, render1, inputs, '1')
            self.render_to_input(label2, render2, inputs, '2')

        # todo inputs
        inputs['ori_img1'] = img_1
        inputs['ori_img2'] = img_2
        inputs['label1'] = label1
        inputs['label2'] = label2
        inputs['aflow'] = aflow

        # need delete
        inputs['idx'] = metadata['idx']

        return inputs

    def crop_img_max_info(self, img_1, img_2, label1, label2, aflow, valid_mask1,
                          depth1, depth2, render1, render2, idx=None):
        # img_1 , img_2 = metadata['img1'], metadata['img2']
        # aflow, label1, valid_mask1 = metadata['aflow'], metadata['label1'], metadata['valid_mask1']
        # label2 = metadata['label2']
        # if 'depth2' in metadata:
        #     depth2 = metadata['depth2']

        crop_size = self.crop({'imsize': (10000, 10000)})[
            'imsize']  # me from RandomCrop(400)
        output_size_a = min(img_1.size, crop_size)
        output_size_b = min(img_2.size, crop_size)
        img_1 = np.array(img_1)
        img_2 = np.array(img_2)

        ah, aw, p1 = img_1.shape
        bh, bw, p2 = img_2.shape

        assert p1 == 3
        assert p2 == 3
        assert aflow.shape == (ah, aw, 2)
        assert valid_mask1.shape == (ah, aw), ipdb.set_trace()
        #show_flow_mask_image(img_1, img_2, valid_mask1, aflow.transpose(2,0,1))
        # Let's start by computing the scale of the
        # optical flow and applying a median filter:
        dx = np.gradient(aflow[:, :, 0])
        dy = np.gradient(aflow[:, :, 1])
        scale = np.sqrt(
            np.clip(np.abs(dx[1]*dy[0] - dx[0]*dy[1]), 1e-16, 1e16))

        accu2 = np.zeros((16, 16), bool)
        def Q(x, w): return np.int32(16 * (x - w.start) / (w.stop - w.start))

        def window1(x, size, w):
            l = x - int(0.5 + size / 2)
            r = l + int(0.5 + size)
            if l < 0:
                l, r = (0, r - l)
            if r > w:
                l, r = (l + w - r, w)
            if l < 0:
                l, r = 0, w  # larger than width
            return slice(l, r)

        def window(cx, cy, win_size, scale, img_shape):
            return (window1(cy, win_size[1]*scale, img_shape[0]),
                    window1(cx, win_size[0]*scale, img_shape[1]))

        n_valid_pixel = valid_mask1.sum()
        sample_w = valid_mask1 / (1e-16 + n_valid_pixel)

        def sample_valid_pixel():
            n = np.random.choice(sample_w.size, p=sample_w.ravel())
            y, x = np.unravel_index(n, sample_w.shape)
            return x, y

        # Find suitable left and right windows
        trials = 0  # take the best out of few trials
        best = -np.inf, None
        for _ in range(50*self.n_samples):
            if trials >= self.n_samples:
                break  # finished!

            # pick a random valid point from the first image
            if n_valid_pixel == 0:
                break
            c1x, c1y = sample_valid_pixel()

            # Find in which position the center of the left
            # window ended up being placed in the right image
            c2x, c2y = (aflow[c1y, c1x] + 0.5).astype(np.int32)
            if not(0 <= c2x < bw and 0 <= c2y < bh):
                continue

            # Get the flow scale
            sigma = scale[c1y, c1x]

            # Determine sampling windows
            if 0.2 < sigma < 1:
                win1 = window(c1x, c1y, output_size_a, 1/sigma, img_1.shape)
                win2 = window(c2x, c2y, output_size_b, 1, img_2.shape)
            elif 1 <= sigma < 5:
                win1 = window(c1x, c1y, output_size_a, 1, img_1.shape)
                win2 = window(c2x, c2y, output_size_b, sigma, img_2.shape)
            else:
                continue  # bad scale

            # compute a score based on the flow
            x2, y2 = aflow[win1].reshape(-1, 2).T.astype(np.int32)
            # Check the proportion of valid flow vectors
            valid = (win2[1].start <= x2) & (x2 < win2[1].stop) \
                & (win2[0].start <= y2) & (y2 < win2[0].stop)
            score1 = (valid * valid_mask1[win1].ravel()).mean()
            # check the coverage of the second window
            accu2[:] = False
            accu2[Q(y2[valid], win2[0]), Q(x2[valid], win2[1])] = True
            score2 = accu2.mean()
            # Check how many hits we got
            score = min(score1, score2)

            trials += 1
            if score > best[0]:
                best = score, win1, win2

        if None in best:  # couldn't find a good window
            print('debug****************************************')
            print('idx:', idx)
            # ipdb.set_trace()
        else:
            win1, win2 = best[1:]

            img_1 = img_1[win1]
            label1 = label1[win1]

            img_2 = img_2[win2]
            label2 = label2[win2]

            aflow = aflow[win1] - \
                np.float32([[[win2[1].start, win2[0].start]]])
            valid_mask1 = valid_mask1[win1]
            aflow[~valid_mask1.view(bool)] = np.nan  # mask bad pixels!
            aflow = aflow.transpose(2, 0, 1)  # --> (2,H,W)

            if self.use_depth:
                depth1 = np.array(depth1)[win1]
                depth2 = np.array(depth2)[win2]

            if self.use_render:
                #! render1
                render1['color'] = render1['color'][:, win1[0].start:win1[0].stop,
                                                    win1[1].start:win1[1].stop]
                render1['label'] = render1['label'][:, win1[0].start:win1[0].stop,
                                                    win1[1].start:win1[1].stop]
                if self.use_depth:
                    render1['depth'] = render1['depth'][:, win1[0].start:win1[0].stop,
                                                        win1[1].start:win1[1].stop]

                #! render2
                for cc in range(self.render_img_num):
                    render2['color'][cc] = np.expand_dims(
                        np.array(render2['color'][cc]), 0)
                    render2['label'][cc] = np.expand_dims(
                        np.array(render2['label'][cc]), 0)

                render_color_2 = np.concatenate(render2['color'])
                render_label_2 = np.concatenate(render2['label'])

                render2['color'] = render_color_2[:, win2[0].start:win2[0].stop,
                                                  win2[1].start:win2[1].stop]
                render2['label'] = render_label_2[:, win2[0].start:win2[0].stop,
                                                  win2[1].start:win2[1].stop]

                if self.use_depth:
                    render2['depth'] = render2['depth'][:, win2[0].start:win2[0].stop,
                                                        win2[1].start:win2[1].stop].astype(np.float32)

            # rescale if necessary
            if img_1.shape[:2][::-1] != output_size_a:
                sx, sy = (np.float32(output_size_a)-1) / \
                    (np.float32(img_1.shape[:2][::-1])-1)
                img_1 = np.array(Image.fromarray(img_1).resize(
                    output_size_a, Image.ANTIALIAS))
                valid_mask1 = np.array(Image.fromarray(
                    valid_mask1).resize(output_size_a, Image.NEAREST))
                label1 = np.array(Image.fromarray(
                    label1).resize(output_size_a, Image.NEAREST))
                if self.use_depth:
                    depth1 = np.array(Image.fromarray(
                        depth1).resize(output_size_a, Image.NEAREST))

                if self.use_render:
                    render_color_1 = np.zeros(
                        (self.render_img_num, output_size_a[0], output_size_a[1], 3))
                    render_label_1 = np.zeros(
                        (self.render_img_num, output_size_a[0], output_size_a[1]))
                    for cc in range(self.render_img_num):
                        render_color_1[cc] = np.array(Image.fromarray(render1['color'][cc]).resize(output_size_a,
                                                                                                   Image.NEAREST))
                        render_label_1[cc] = np.array(Image.fromarray(render1['label'][cc]).resize(output_size_a,
                                                                                                   Image.NEAREST))
                    render1['color'] = render_color_1
                    render1['label'] = render_label_1
                    if self.use_depth:
                        render_depth_1 = np.zeros(
                            (self.render_img_num, output_size_a[0], output_size_a[1]))
                        for cc in range(self.render_img_num):
                            render_depth_1[cc] = np.array(Image.fromarray(render1['depth'][cc]).resize(output_size_a,
                                                                                                       Image.NEAREST))
                        render1['depth'] = render_depth_1

                afx = Image.fromarray(aflow[0]).resize(
                    output_size_a, Image.NEAREST)
                afy = Image.fromarray(aflow[1]).resize(
                    output_size_a, Image.NEAREST)
                aflow = np.stack((np.float32(afx), np.float32(afy)))

            if img_2.shape[:2][::-1] != output_size_b:
                sx, sy = (np.float32(output_size_b)-1) / \
                    (np.float32(img_2.shape[:2][::-1])-1)
                img_2 = np.array(Image.fromarray(img_2).resize(
                    output_size_b, Image.ANTIALIAS))
                label2 = np.array(Image.fromarray(
                    label2).resize(output_size_b, Image.NEAREST))
                aflow *= [[[sx]], [[sy]]]

                if self.use_depth:
                    depth2 = np.array(Image.fromarray(
                        depth2).resize(output_size_b, Image.NEAREST))

                if self.use_render:
                    render_color_2 = np.zeros(
                        (self.render_img_num, output_size_b[0], output_size_b[1], 3))
                    render_label_2 = np.zeros(
                        (self.render_img_num, output_size_b[0], output_size_b[1]))
                    for cc in range(self.render_img_num):
                        render_color_2[cc] = np.array(Image.fromarray(render2['color'][cc]).resize(output_size_b,
                                                                                                   Image.NEAREST))
                        render_label_2[cc] = np.array(Image.fromarray(render2['label'][cc]).resize(output_size_b,
                                                                                                   Image.NEAREST))
                    render2['color'] = render_color_2
                    render2['label'] = render_label_2

                    if self.use_depth:
                        render_depth_2 = np.zeros(
                            (self.render_img_num, output_size_b[0], output_size_b[1]))
                        for cc in range(self.render_img_num):
                            render_depth_2[cc] = np.array(Image.fromarray(render2['depth'][cc]).resize(output_size_b,
                                                                                                       Image.NEAREST))
                        render2['depth'] = render_depth_2.astype(np.float32)

        return img_1, img_2, label1, label2, aflow, valid_mask1, depth1, depth2, render1, render2

    def render_to_input(self, labelx, renderx, inputs, x):
        labelx_modify = np.where(labelx > 0, labelx, -1)
        labelx_modify = np.expand_dims(labelx_modify, 0).repeat(
            renderx['color'].shape[0], 0)
        render_valid_x = np.where(
            renderx['label'] == labelx_modify, True, False)

        inputs['render_valid_' + x] = np.expand_dims(render_valid_x, 1)
        inputs['render_color_' + x] = torch.from_numpy(
            (renderx['color'].astype(np.float32)/255.0 - 0.45)/0.225).permute(0, 3, 1, 2)
        if self.use_depth:
            inputs['render_depth_' +
                   x] = self.depth_norm(renderx['depth'] / 65535)
