import ipdb
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageEnhance
import cv2
from scipy.spatial import ConvexHull, distance_matrix

from core.tools.depth_map_utils_ycb import fill_in_multiscale, fill_in_fast

# implemented by r2d2: https://github.com/naver/r2d2/blob/master/tools/transforms_tools.py

class DummyImg:
    ''' This class is a dummy image only defined by its size.
    '''

    def __init__(self, size):
        self.size = size

    def resize(self, size, *args, **kwargs):
        return DummyImg(size)

    def expand(self, border):
        w, h = self.size
        if isinstance(border, int):
            size = (w+2*border, h+2*border)
        else:
            l, t, r, b = border
            size = (w+l+r, h+t+b)
        return DummyImg(size)

    def crop(self, border):
        w, h = self.size
        l, t, r, b = border
        assert 0 <= l <= r <= w
        assert 0 <= t <= b <= h
        size = (r-l, b-t)
        return DummyImg(size)

    def rotate(self, angle):
        raise NotImplementedError

    def transform(self, size, *args, **kwargs):
        return DummyImg(size)


def grab_img(img_and_label):
    ''' Called to extract the image from an img_and_label input
    (a dictionary). Also compatible with old-style PIL images.
    '''
    if isinstance(img_and_label, dict):
        # if input is a dictionary, then
        # it must contains the img or its size.
        try:
            return img_and_label['img']
        except KeyError:
            return DummyImg(img_and_label['imsize'])

    else:
        # or it must be the img directly
        return img_and_label


def update_img_and_labels(img_and_label, img, persp=None, persp2=None, theta=None):
    ''' Called to update the img_and_label
    '''
    if isinstance(img_and_label, dict):
        img_and_label['img'] = img
        img_and_label['imsize'] = img.size

        if persp:
            if 'persp' not in img_and_label:
                img_and_label['persp'] = (1, 0, 0, 0, 1, 0, 0, 0)
            img_and_label['persp'] = persp_mul(persp, img_and_label['persp'])

        if persp2:
            img_and_label['persp2'] = persp2

        if theta:
            img_and_label['theta'] = theta

        return img_and_label

    else:
        # or it must be the img directly
        return img


def rand_log_uniform(a, b):
    return np.exp(np.random.uniform(np.log(a), np.log(b)))


def translate(tx, ty):
    return (1, 0, tx,
            0, 1, ty,
            0, 0)


def rotate(angle):
    return (np.cos(angle), -np.sin(angle), 0,
            np.sin(angle), np.cos(angle), 0,
            0, 0)


def persp_mul(mat, mat2):
    ''' homography (perspective) multiplication.
    mat: 8-tuple (homography transform)
    mat2: 8-tuple (homography transform) or 2-tuple (point)
    '''
    assert isinstance(mat, tuple)
    assert isinstance(mat2, tuple)

    mat = np.float32(mat+(1,)).reshape(3, 3)
    mat2 = np.array(mat2+(1,)).reshape(3, 3)
    res = np.dot(mat, mat2)
    return tuple((res/res[2, 2]).ravel()[:8])


def persp_apply(mat, pts):
    ''' homography (perspective) transformation.
    mat: 8-tuple (homography transform)
    pts: numpy array
    '''
    assert isinstance(mat, tuple)
    assert isinstance(pts, np.ndarray)
    assert pts.shape[-1] == 2
    mat = np.float32(mat+(1,)).reshape(3, 3)

    if pts.ndim == 1:
        pt = np.dot(pts, mat[:, :2].T).ravel() + mat[:, 2]
        pt /= pt[2]  # homogeneous coordinates
        return tuple(pt[:2])
    else:
        pt = np.dot(pts, mat[:, :2].T) + mat[:, 2]
        pt[:, :2] /= pt[:, 2:3]  # homogeneous coordinates
        return pt[:, :2]


def is_pil_image(img):
    return isinstance(img, Image.Image)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    brightness_factor (float):  How much to adjust the brightness. Can be
    any non negative number. 0 gives a black image, 1 gives the
    original image while 2 increases the brightness by a factor of 2.
    Returns:
    PIL Image: Brightness adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    contrast_factor (float): How much to adjust the contrast. Can be any
    non negative number. 0 gives a solid gray image, 1 gives the
    original image while 2 increases the contrast by a factor of 2.
    Returns:
    PIL Image: Contrast adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    saturation_factor (float):  How much to adjust the saturation. 0 will
    give a black and white image, 1 will give the original image while
    2 will enhance the saturation by a factor of 2.
    Returns:
    PIL Image: Saturation adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    hue_factor (float):  How much to shift the hue channel. Should be in
    [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
    HSV space in positive and negative direction respectively.
    0 means no shift. Therefore, both -0.5 and 0.5 will give an image
    with complementary colors while 0 gives the original image.
    Returns:
    PIL Image: Hue adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError(
            'hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
        h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def img_coord_2_obj_coord(kp2d, depth, k, pose_obj2cam, retain_3d=False):
    # kp2d (num, 3)
    # inv_k (3, 3)
    # depth (num,)
    # pose_obj2cam (4, 4)

    #depth = depth
    inv_k = np.linalg.inv(k[:3, :3])
    #pose_obj2cam = pose_obj2cam
    kp2d = kp2d[:, :2]
    kp2d = np.concatenate((kp2d, np.ones((kp2d.shape[0], 1))), 1)

    kp2d_int = np.round(kp2d).astype(np.int)[:, :2]
    kp_depth = depth[kp2d_int[:, 1], kp2d_int[:, 0]]  # num

    kp2d_cam = np.expand_dims(kp_depth, 1) * kp2d  # num, 3
    kp3d_cam = np.dot(inv_k, kp2d_cam.T).T  # num, 3

    kp3d_cam_pad1 = np.concatenate(
        (kp3d_cam, np.ones((kp2d_cam.shape[0], 1))), 1).T  # 4, num
    kp3d_obj = np.dot(np.linalg.inv(pose_obj2cam), kp3d_cam_pad1).T  # num, 4

    if retain_3d:
        return kp3d_obj[:, :3]
    else:
        return kp3d_obj


def img_coord_2_cam_coord(kp2d, depth, inv_k):
    # kp2d (num, 3)
    # inv_k (3, 3)
    # depth (num,)
    kp2d_cam = np.dot(inv_k, kp2d.t()).t()  # num, 3
    kp3d_cam = depth * kp2d_cam  # num, 3

    return kp3d_cam


def project_p3d(p3d, cam_scale, K):
    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d


def tensor2img(tensor, model=None):
    """ convert back a torch/numpy tensor to a PIL Image
        by undoing the ToTensor() and Normalize() transforms.
    """
    RGB_mean = [0.485, 0.456, 0.406]
    RGB_std = [0.229, 0.224, 0.225]

    norm_RGB = tvf.Compose(
        [tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])
    mean = norm_RGB.transforms[1].mean
    std = norm_RGB.transforms[1].std
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    res = np.uint8(
        np.clip(255*((tensor.transpose(1, 2, 0) * std) + mean), 0, 255))
    from PIL import Image
    return Image.fromarray(res)


def fill_missing(
    dpt, cam_scale, scale_2_80m, fill_type='multiscale',
    extrapolate=False, show_process=False, blur_type='bilateral'
):
    dpt = dpt / cam_scale * scale_2_80m
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt / scale_2_80m * cam_scale
    return dpt


def inv_pose(ori_pose):
    ipdb.set_trace()
    if torch.is_tensor(ori_pose):
        if ori_pose.shape[0] == 3:
            ori_pose_new = torch.cat(
                (ori_pose, torch.array([[0, 0, 0, 1]])), axis=0)
            return torch.inverse(ori_pose_new)[:3]

        return torch.inverse(ori_pose)

    else:
        if ori_pose.shape[0] == 3:
            ori_pose_new = np.concatenate(
                (ori_pose, np.array([[0, 0, 0, 1]])), axis=0)
            return np.linalg.inv(ori_pose_new)[:3]

        return np.linalg.inv(ori_pose)


def compute_bbox(pose, K, scale_size=230, scale=(1, 1, 1)):
    obj_x = pose[0, 3] * scale[0]
    obj_y = pose[1, 3] * scale[1]
    obj_z = pose[2, 3] * scale[2]
    offset = scale_size / 2
    points = np.ndarray((4, 3), dtype=np.float)
    points[0] = [obj_x - offset, obj_y - offset, obj_z]     # top left
    points[1] = [obj_x - offset, obj_y + offset, obj_z]     # top right
    points[2] = [obj_x + offset, obj_y - offset, obj_z]     # bottom left
    points[3] = [obj_x + offset, obj_y + offset, obj_z]     # bottom right
    projected_vus = np.zeros((points.shape[0], 2))
    projected_vus[:, 1] = points[:, 0] * K[0, 0] / points[:, 2] + K[0, 2]
    projected_vus[:, 0] = points[:, 1] * K[1, 1] / points[:, 2] + K[1, 2]
    projected_vus = np.round(projected_vus).astype(np.int32)
    return projected_vus


def crop_bbox(color, depth, boundingbox, output_size=(100, 100), seg=None):
    left = np.min(boundingbox[:, 1])
    right = np.max(boundingbox[:, 1])
    top = np.min(boundingbox[:, 0])
    bottom = np.max(boundingbox[:, 0])

    h, w, c = color.shape
    crop_w = right - left
    crop_h = bottom - top
    color_crop = np.zeros((crop_h, crop_w, 3), dtype=color.dtype)
    depth_crop = np.zeros((crop_h, crop_w), dtype=np.float)
    seg_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
    top_offset = abs(min(top, 0))
    bottom_offset = min(crop_h - (bottom - h), crop_h)
    right_offset = min(crop_w - (right - w), crop_w)
    left_offset = abs(min(left, 0))

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, h)
    right = min(right, w)
    color_crop[top_offset:bottom_offset, left_offset:right_offset,
               :] = color[top:bottom, left:right, :]
    depth_crop[top_offset:bottom_offset,
               left_offset:right_offset] = depth[top:bottom, left:right]
    resized_rgb = cv2.resize(color_crop, output_size,
                             interpolation=cv2.INTER_NEAREST)
    resized_depth = cv2.resize(
        depth_crop, output_size, interpolation=cv2.INTER_NEAREST)

    if seg is not None:
        seg_crop[top_offset:bottom_offset,
                 left_offset:right_offset] = seg[top:bottom, left:right]
        resized_seg = cv2.resize(
            seg_crop, output_size, interpolation=cv2.INTER_NEAREST)
        final_seg = resized_seg.copy()

    mask_rgb = resized_rgb != 0
    mask_depth = resized_depth != 0
    resized_depth = resized_depth.astype(np.uint16)
    final_rgb = resized_rgb * mask_rgb
    final_depth = resized_depth * mask_depth
    if seg is not None:
        return final_rgb, final_depth, final_seg
    else:
        return final_rgb, final_depth


def compute_cloud_diameter(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    distances = distance_matrix(hull_points, hull_points)
    return np.max(distances)


def desc_split(desc, obj_dims):
    if obj_dims > 0:
        desc_distinct = desc[:, :-obj_dims]
        desc_obj = desc[:, -obj_dims:]
    else:
        desc_distinct = desc
        desc_obj = None

    return desc_distinct, desc_obj


def camera_intrisinc_tile(K, to_tensor=True):
    '''
    input:
    K: 3*3 mat
    '''
    if K.shape[0] == 3:
        camera_intr_44 = np.eye(4, dtype=np.float32)
        camera_intr_44[:3, :3] = K

    camera_intr_inv_44 = np.linalg.inv(camera_intr_44)
    if to_tensor:
        camera_intr_44 = torch.tensor(np.expand_dims(
            camera_intr_44, 0))  # , dtype=torch.double
        camera_intr_inv_44 = torch.tensor(np.expand_dims(
            camera_intr_inv_44, 0))  # , dtype=torch.double

    return camera_intr_44, camera_intr_inv_44
