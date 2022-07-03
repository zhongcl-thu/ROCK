import os
import ipdb
import numpy as np
from PIL import Image
import scipy.io as scio

import torch
import torchvision.transforms as tvf
from torch.utils.data import Dataset

from core.tools.transforms_tools import fill_missing
from core.tools.viz import show_mask_image, show_depth_image


class SingleDataset (Dataset):
    def __init__(self, config, filenames, test_type='sim2real_6dpose'):
        Dataset.__init__(self)

        data_info = config["data_info"]

        self.need_read = data_info['need_read']
        self.use_depth = False if 'depth' not in self.need_read else True

        self.aug_params = config['augmentation']
        self.norm_rgb = tvf.Compose([tvf.ToTensor(),
                                     tvf.Normalize(mean=self.aug_params['rgb_norm_mean'],
                                                   std=self.aug_params['rgb_norm_std'])])
        self.depth_norm = tvf.Normalize(mean=self.aug_params['depth_norm_mean'],
                                        std=self.aug_params['depth_norm_std'])

        self.test_info = config["test"][test_type]
        self.root = self.test_info['test_img_root']

        self.filenames = filenames
        self.data_info = data_info

    def __len__(self):
        return len(self.filenames)

    def get_key(self, idx, suffix):
        return self.filenames[idx] + suffix

    def get_pose(self, index):
        folder_name, img_name = self.filenames[index].split(' ')
        j = '%04d' % int(img_name)
        pose_dict = np.load(self.x(folder_name+'/' + j +
                            '-pose.npy'), allow_pickle=True).item()
        Tco = pose_dict['obj_pose']
        pos_id = pose_dict['pose_id']
        cam_pose = pose_dict['cam_pose']
        K = pose_dict['intrins']

        return Tco, pos_id, K, cam_pose

    def get_render_image(self, idx, pair_index, use_depth=False):
        file_name, img_name = self.filenames[idx].split(' ')
        file_name = '%03d' % int(file_name)
        img_name = '%04d' % int(img_name)

        render_name = os.path.join(self.file_root, file_name, 'obj_index.txt')
        obj_index = np.loadtxt(render_name, dtype=np.int16)

        render_color = []
        render_mask = []
        render_depth = []

        for cc in obj_index:
            fname_color = os.path.join(
                self.file_root, file_name,
                img_name+'-'+pair_index+'-'+str(int(cc))+'-color-render.png')
            fname_mask = os.path.join(
                self.file_root, file_name,
                img_name+'-'+pair_index+'-'+str(int(cc))+'-mask-render.png')
            try:
                img = np.array(Image.open(fname_color).convert('RGB'))
                mask = np.array(Image.open(fname_mask))
                render_color.append(np.expand_dims(img, 0))
                render_mask.append(np.expand_dims(mask, 0))
            except Exception as e:
                raise IOError("Could not load image %s (reason: %s)" %
                              (fname_color, str(e)))

            if use_depth:
                fname_depth = os.path.join(
                    self.file_root, file_name,
                    img_name+'-'+pair_index+'-'+str(cc)+'-depth-render.png')
                try:
                    depth = np.array(Image.open(fname_depth))
                    render_depth.append(np.expand_dims(depth, 0))
                except Exception as e:
                    raise IOError("Could not load image %s (reason: %s)" % (
                        fname_depth, str(e)))

        return np.concatenate(render_color), np.concatenate(render_mask), \
            np.concatenate(render_depth) if use_depth else None

    def __getitem__(self, idx):
        raise NotImplementedError()


class Real_YCB_Dataset(SingleDataset):
    def get_key(self, idx, suffix):
        folder_name, img_name = self.filenames[idx].split('/')
        return os.path.join(folder_name, img_name + suffix)

    def __getitem__(self, index):
        inputs = {}

        img_name = os.path.join(self.root, self.get_key(index, '-color.png'))
        inputs['ori_img'] = np.array(Image.open(img_name))
        inputs['img'] = self.norm_rgb(inputs['ori_img'])

        label_name1 = os.path.join(
            self.root, self.get_key(index, '-label.png'))
        inputs['label'] = np.array(Image.open(label_name1))

        box_name = os.path.join(self.root, self.get_key(index, '-box.txt'))

        meta_name = os.path.join(self.root, self.get_key(index, '-meta.mat'))
        meta = scio.loadmat(meta_name)

        '''pre-process_depth'''
        depth_name = os.path.join(self.root, self.get_key(index, '-depth.png'))
        cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
        dpt = np.array(Image.open(depth_name), dtype=np.float32)
        dpt = fill_missing(dpt, cam_scale, 1) / cam_scale
        inputs['depth'] = dpt
        valid_mask = np.logical_and(dpt >= self.test_info['min_depth'],
                                    dpt <= self.test_info['max_depth'])
        inputs['ori_img'] *= np.expand_dims(valid_mask, 2)
        # show_depth_image(inputs['ori_img1'], None, dpt, None, save=False)
        # show_depth_image(inputs['ori_img1']*np.expand_dims(valid_mask,2),
        #                   None, dpt*valid_mask, None, save=False)
        # ipdb.set_trace()

        if self.use_depth:
            dpt_tensor = tvf.ToTensor()(dpt.copy())

            if self.aug_params['depth_normalize'] == 'variable':
                dpt_tensor = (dpt_tensor-dpt_tensor.min()) / \
                    (dpt_tensor.max()-dpt_tensor.min())
            else:
                dpt_tensor /= 65535

            dpt_tensor = self.depth_norm(dpt_tensor)
            inputs['img'] = torch.cat(
                (inputs['img'], dpt_tensor.repeat(3, 1, 1)), 0)

        inputs['valid_mask'] = torch.from_numpy(valid_mask).unsqueeze(0)
        inputs['img'] *= inputs['valid_mask']

        if 'obj_pose' in self.need_read:
            inputs['pose_id'] = meta['cls_indexes']
            poses = meta['poses'].transpose(2, 0, 1)
            ones = np.array([[[0, 0, 0, 1.0]]]).repeat(poses.shape[0], axis=0)
            inputs['cam_obj_poses'] = np.concatenate((poses, ones), 1)

            # bounding box
            with open(box_name, 'r') as f:
                boxes = f.read().splitlines()
            boxes_dict = {}
            for i in range(len(boxes)):
                box_i = boxes[i].split(' ')
                cc = int(inputs['pose_id'][i])
                boxes_dict[cc] = np.int16(np.float32(box_i[1:]))
            inputs['box'] = boxes_dict

        inputs['K'] = meta['intrinsic_matrix']
        #inputs['cam_pose1'] = np.concatenate((meta['rotation_translation_matrix'], ones[0]), 0)
        inputs['cam_pose'] = np.linalg.inv(np.concatenate((meta['rotation_translation_matrix'],
                                                           ones[0]), 0))

        return inputs


class YCBInEOAT_Dataset(SingleDataset):
    def get_key(self, idx, suffix):
        folder_name, img_name = self.filenames[idx].split(' ')
        return os.path.join(folder_name, suffix, img_name+'.png')

    def __getitem__(self, index):
        inputs = {}
        inputs['K'] = np.array([[319.58200073242, 0.0, 320.2149847676950, 0.0],
                                [0.0, 417.118682861, 244.3486680871, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])

        img_name = os.path.join(self.root, self.get_key(index, 'rgb'))
        inputs['ori_img'] = np.array(Image.open(img_name).convert("RGB"))
        inputs['img'] = self.norm_rgb(inputs['ori_img'])

        # author's mask
        # label_name1 = os.path.join(self.root, self.get_key(index, 'gt_mask'))
        folder_name, img_name = self.filenames[index].split(' ')
        label_name1 = os.path.join(
            self.root, folder_name, 'label_me', 'label.png')
        inputs['label'] = np.array(Image.open(label_name1))

        '''pre-process_depth'''
        depth_name = os.path.join(
            self.root, self.get_key(index, 'depth_filled'))
        dpt = np.array(Image.open(depth_name), dtype=np.float32) / 1000

        inputs['depth'] = dpt
        valid_mask = np.logical_and(
            dpt >= self.test_info['min_depth'], dpt <= 10)
        inputs['ori_img'] *= np.expand_dims(valid_mask, 2)

        if self.use_depth:
            dpt_tensor = tvf.ToTensor()(dpt.copy())
            if self.depth_max_min_normalize:
                dpt_tensor = (dpt_tensor - dpt_tensor.min()) / \
                    (dpt_tensor.max() - dpt_tensor.min())
            else:
                dpt_tensor /= self.max_depth

            dpt_tensor = (dpt_tensor - 0.45) / 0.225

            inputs['img'] = torch.cat(
                (inputs['img'], dpt_tensor.repeat(3, 1, 1)), 0)

        inputs['valid_mask'] = torch.from_numpy(valid_mask).unsqueeze(0)
        inputs['img'] = inputs['valid_mask'] * inputs['img']

        if 'obj_pose' in self.need_read:
            obj_pose_name = os.path.join(self.root, self.get_key(
                index, 'annotated_poses')).replace('.png', '.txt')
            inputs['cam_obj_poses'] = np.float32(
                np.loadtxt(obj_pose_name))  # 4*4

        return inputs


class Render_templates_Dataset(SingleDataset):
    def __init__(self, config, filenames, category):
        super(Render_templates_Dataset, self).__init__(config, filenames)

        self.root = self.test_info['ref_img_root']
        self.category = category

    def get_key(self, idx, suffix):
        img_name = '{:0>3d}'.format(idx)
        return img_name + suffix

    def __getitem__(self, index):
        inputs = {}

        root_template_path = os.path.join(self.root,
                                          str(self.test_info['ref_img_num']),
                                          '{:0>3d}'.format(self.category))

        img_name = os.path.join(root_template_path,
                                self.get_key(index, '-color.png'))
        inputs['ori_img'] = np.array(Image.open(img_name).convert('RGB'))
        inputs['img'] = self.norm_rgb(inputs['ori_img'])

        label_name = os.path.join(root_template_path,
                                  self.get_key(index, '-mask.png'))
        inputs['label'] = np.array(Image.open(label_name))

        valid_mask = np.where(inputs['label'] > 0, 1.0, 0.0)

        '''pre-process_depth'''
        depth_name = os.path.join(
            root_template_path, self.get_key(index, '-depth.png'))
        dpt = np.array(Image.open(depth_name), dtype=np.float32)
        inputs['depth'] = dpt / 65535 * 2.0

        if self.use_depth:
            dpt_tensor = tvf.ToTensor()(dpt.copy())

            if self.aug_params['depth_normalize'] == 'variable':
                dpt_tensor = (dpt_tensor-dpt_tensor.min()) / \
                    (dpt_tensor.max()-dpt_tensor.min())
            else:
                dpt_tensor /= 65535

            dpt_tensor = self.depth_norm(dpt_tensor)
            inputs['img'] = torch.cat(
                (inputs['img'], dpt_tensor.repeat(3, 1, 1)), 0)

        inputs['valid_mask'] = torch.from_numpy(valid_mask).unsqueeze(0)
        #inputs['img'] *= inputs['valid_mask']

        intrins_i = np.loadtxt(os.path.join(self.root, str(
            self.test_info['ref_img_num']), 'intrincs.txt'))
        camera_intr_44 = np.eye(4, dtype=np.float32)
        camera_intr_44[:3, :3] = intrins_i
        inputs['K'] = camera_intr_44

        cam_obj_pose = np.load(os.path.join(self.root, str(
            self.test_info['ref_img_num']), 'obj_pose.npy'))
        inputs['cam_obj_poses'] = cam_obj_pose[index]

        return inputs
