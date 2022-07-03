import os
import ipdb
import tqdm
import json
import glob
import copy
import shutil
from easydict import EasyDict as edict

import numpy as np
import torch
from torch.utils.data import DataLoader

import core.nets as nets
from core.nets.sampler import FullSampler
from core.datasets import *

from core.utils import gen_pix_coords, store_json, load_model
from core.tools.match import *
from core.tools.transforms_tools import desc_split, img_coord_2_obj_coord, camera_intrisinc_tile
from core.tools.viz import draw_matches, draw_matches_v2
from core.tools.metric import cal_auc

from core.test.test_utils import *


class Evaluator:
    def __init__(self, C):
        self.C = C
        self.config_common = edict(C.config["common"])
        self.config_test = edict(C.config["test"])
        self.data_info = edict(C.config["data_info"])
        self.config_evaluate = edict(C.config['evaluate'])

        self.model_path = self.data_info.model_path
        self.models_to_load = self.data_info.models_to_load

        self.use_relia_score = self.config_common.public_params.use_relia_score

        self.detector = NonMaxSuppression(
            base_thr=self.config_evaluate.reliability_thr,
            refine_thr=self.config_evaluate.repeatability_thr)

        self.test_obj_dim = self.config_test.test_obj_dim

        save_path = self.config_common.get(
            "save_path", os.path.dirname(C.config_file))
        self.tmp_template_desc_path = "{}/results/tmp_ref_img_desc".format(
            save_path)
        self.pose_results_path = "{}/results/6d_pose".format(save_path)
        self.track_results_path = "{}/results/track".format(save_path)
        self.sim2real_match_results_path = "{}/results/sim2real_match".format(
            save_path)

        if not os.path.exists(self.tmp_template_desc_path):
            os.makedirs(self.tmp_template_desc_path)
        else:
            shutil.rmtree(self.tmp_template_desc_path)
            os.makedirs(self.tmp_template_desc_path)
        if not os.path.exists(self.pose_results_path):
            os.makedirs(self.pose_results_path)
        if not os.path.exists(self.track_results_path):
            os.makedirs(self.track_results_path)
        if not os.path.exists(self.sim2real_match_results_path):
            os.makedirs(self.sim2real_match_results_path)

    def initialize(self, args):
        self.device = args.device
        self.multi_gpu = args.multi_gpu
        self.local_rank = args.local_rank
        self.test_type = args.test_type
        self.test_model_root = args.test_model_root
        self.nprocs = torch.cuda.device_count()

        self.create_model()
        self.create_dataset()
        self.create_dataloader()

    def create_model(self):
        self.model = nets.model_entry(self.config_common.net,
                                      self.config_common.public_params)

        load_model(os.path.join(self.test_model_root, self.model_path),
                   self.models_to_load, self.model.net)

        for k, m in self.model.net.items():
            if self.config_test.set_to_eval:
                m.to(self.device).eval()
            else:
                m.to(self.device)
            if self.multi_gpu:
                self.model.net[k] = self.set_model_ddp(m)

    def create_dataset(self,):
        if self.test_type == 'sim2real_6dpose':
            self.test_dataset = {}
            for vi in self.config_test[self.test_type]['videos']:
                with open(self.config_test.sim2real_6dpose.test_file_root.format(vi), 'r') as f:
                    v_f = f.read().splitlines()

                self.test_dataset[vi] = Real_YCB_Dataset(
                    self.C.config, v_f, self.test_type)

            self.template_dataset = {}
            for category in range(self.config_common.public_params.num_class):
                self.template_dataset[category] = Render_templates_Dataset(config=self.C.config,
                                                                           filenames=range(
                                                                               self.config_test.sim2real_6dpose.ref_img_num),
                                                                           category=category)

        elif self.test_type == 'match_tracking':
            self.test_dataset = {}
            for vi in self.config_test.match_tracking.videos:
                with open(self.config_test.match_tracking.test_file_root.format(vi), 'r') as f:
                    v_f = f.read().splitlines()

                self.test_dataset[vi] = YCBInEOAT_Dataset(
                    self.C.config, v_f, self.test_type)

        elif self.test_type == 'sim2real_match':
            self.test_dataset = {}
            for vi in self.config_test.sim2real_match.videos:
                with open(self.config_test.sim2real_match.test_file_root.format(vi), 'r') as f:
                    v_f = f.read().splitlines()
                    self.test_dataset[vi] = Real_YCB_Dataset(
                        self.C.config, v_f, self.test_type)
        else:
            raise NotImplemented

    def create_dataloader(self):
        self.test_loader = {}
        for vi in self.config_test[self.test_type]['videos']:
            self.test_loader[vi] = DataLoader(self.test_dataset[vi],
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=self.config_common.workers,
                                              sampler=None,
                                              pin_memory=True,
                                              drop_last=False
                                              )
        if self.test_type == 'sim2real_6dpose':
            self.template_loader = {}
            for category in range(self.config_common.public_params.num_class):
                self.template_loader[category] = DataLoader(self.template_dataset[category],
                                                            batch_size=self.config_common.batch_size,
                                                            shuffle=False,
                                                            num_workers=self.config_common.workers,
                                                            sampler=None,
                                                            pin_memory=True,
                                                            drop_last=False
                                                            )

    def instantiate_render(self, inputs, renderer_info, classes=None):
        if classes is None:
            self.obj_renderers = {}
            self.object_width_dict = {}
            self.cls_ptsxyz_dict = {}

            self.prepare_data(inputs)
            cam_obj_poses = inputs['cam_obj_poses']
            pose_id = inputs['pose_id']
            classes = inputs['classes']
            cam_pose = inputs['cam_pose']

            self.renderer_res = renderer_info['resolution']

            _, _, self.H, self.W = inputs['img'].shape
            self.render_K = inputs['K'] / (self.H/self.renderer_res[0])
            self.render_K[2, 2] = 1.0  # shape [4, 4]

            self.camera_intr_44, self.camera_intr_inv_44 = camera_intrisinc_tile(
                inputs['K'], to_tensor=True)
            self.bd = BackprojectDepth(1, self.H, self.W)
            self.p3d = Project3D(1, self.H, self.W)
            self.pix_coords = gen_pix_coords(1, self.H, self.W)
        else:
            cam_obj_poses = None
            pose_id = None
            cam_pose = None

            for cc in classes:
                if cc == 0:
                    continue
                render_obj_path = renderer_info[cc, 'obj_path']
                scale = 1.0

                np.loadtxt(renderer_info[(cc, 'pts_path')],
                           dtype=np.float32) * scale  # [2621, 3]

                # instantial renderer class
                self.obj_renderers[cc] = Renderer([render_obj_path], self.render_K,
                                                  H=self.renderer_res[0], W=self.renderer_res[1], scale=scale)  # cam_height, cam_width
                self.object_width_dict[cc] = 0.1

        return cam_obj_poses, pose_id, classes, cam_pose

    def combine_all_obj_add(self,):
        obj_add_result = {}
        for i in range(1, self.config_common.public_params.num_class):
            obj_add_result[(i, 'add')] = []
            obj_add_result[(i, 'adds')] = []

        for name in glob.glob(self.pose_results_path + '/*_result.json'):
            with open(name, 'r') as f:
                video_result = json.loads(f.read())
                for seq, result in video_result.items():
                    for cc, v in result.items():
                        obj_add_result[(int(cc), 'add')].append(float(v[0]))
                        obj_add_result[(int(cc), 'adds')].append(float(v[1]))

        cls_add_auc = []
        cls_adds_auc = []
        total_add_auc = []
        total_adds_auc = []

        for cls_id in range(1, self.config_common.public_params.num_class):
            cls_add_auc.append(cal_auc(obj_add_result[(cls_id, 'add')]))
            cls_adds_auc.append(cal_auc(obj_add_result[(cls_id, 'adds')]))

            total_add_auc.append(obj_add_result[(cls_id, 'add')])
            total_adds_auc.append(obj_add_result[(cls_id, 'adds')])

        total_add_result = cal_auc(np.concatenate(total_add_auc))
        total_adds_result = cal_auc(np.concatenate(total_adds_auc))

        result_info = dict(
            adds_auc=cls_adds_auc,
            add_auc=cls_add_auc,
            total_add=total_add_result,
            total_adds=total_adds_result
        )

        save_path = os.path.join(self.pose_results_path, 'auc_result.json')
        with open(save_path, 'w') as f:
            json.dump(result_info, f, indent=2)

    def prepare_data(self, ins):
        ins['cam_obj_poses'] = ins['cam_obj_poses'][0].numpy()
        ins['classes'] = torch.unique(ins['label']).numpy()
        ins['K'] = ins['K'][0].numpy()

        ins['cam_pose'] = ins.get('cam_pose', torch.eye(4)).numpy()
        ins['pose_id'] = ins.get('pose_id', torch.zeros(4))[0].numpy()
        if 'box' in ins:
            for _, box in ins['box'].items():
                box = box.numpy()

        ins['valid_mask'] = ins['valid_mask'].to(self.device)
        ins['img'] = ins['img'].to(self.device)

    def get_corr_grid(self, inputs):
        aflow = inputs['aflow'].to(self.device)
        label = inputs['label'].to(self.device)

        B, two, H, W = aflow.shape
        assert two == 2

        grid = FullSampler._aflow_to_grid(aflow)
        border_mask = torch.where(grid.abs() <= 1, True, False)
        border_mask = border_mask[:, :, :, 0] * border_mask[:, :, :, 1]
        valid_mask = torch.where(label > 0, True, False)
        valid_mask *= border_mask

        inputs['valid_mask'] = valid_mask
        inputs['grid'] = grid

    def extract_kps_descpritors(self, ins, multi_scale=False, dense_predict=False):
        if dense_predict:
            relia, repreat, desc = extract_single_scale(self.model.net, ins['img'],
                                                        self.detector,
                                                        kp_type='ours', un=True,
                                                        select=False,
                                                        use_relia_score=self.use_relia_score)
            desc_intra, desc_inter = desc_split(desc, self.test_obj_dim)
            return relia, repreat, desc_intra, desc_inter, None
        else:
            xy, desc, _ = extract_single_scale(self.model.net, ins['img'],
                                               self.detector,
                                               kp_type='ours', un=True,
                                               select=True, valid_mask=ins['valid_mask'],
                                               selectnum=int(
                                                   self.config_test.top_k/2),
                                               use_relia_score=self.use_relia_score)
            return None, None, desc, None, xy

    def template_data_inference(self, category):
        tmp_desc_path = os.path.join(self.tmp_template_desc_path,
                                     '{:0>3d}_kps_desc.pth'.format(category))
        if os.path.exists(tmp_desc_path):
            return torch.load(tmp_desc_path)
        else:
            xy_cc_lists_ref = []
            desc_cc_lists_ref = []

            for k, m in self.model.net.items():
                m.eval()

            for temp_inputs in self.template_loader[category]:

                self.prepare_data(temp_inputs)
                #! todo multiscale
                _, _, desc_cc_list_ref, _, xy_cc_list_ref = self.extract_kps_descpritors(temp_inputs,
                                                                                         multi_scale=self.config_test[
                                                                                             self.test_type]['ref_img_multi_scale'],
                                                                                         dense_predict=False)

                xy_cc_lists_ref.extend(xy_cc_list_ref)
                desc_cc_lists_ref.extend(desc_cc_list_ref)

            state = {'kps': xy_cc_lists_ref, 'descs': desc_cc_lists_ref}

            torch.save(state, tmp_desc_path)

            return state

    def sim2real_match(self, metric='mean_mma'):
        # metric=['mean_mma', 'total_mma']
        # some metric and data container
        class_range = range(1, self.config_common.public_params.num_class)
        match_dist = {i: np.array([]) for i in class_range}
        match_kps = {i: 0 for i in class_range}
        class_freq = {i: 0 for i in class_range}
        mma5s = {i: [] for i in class_range}
        mma7s = {i: [] for i in class_range}

        config_sim2real = self.config_test['sim2real_match']
        renderer_info = get_renderer_info(
            config_sim2real['ref_model_path'], classes=class_range)

        for video_num, data_loader_i in self.test_loader.items():
            if not os.path.exists(self.sim2real_match_results_path+'/visualize/video_{:0>2d}'.format(video_num)):
                os.makedirs(self.sim2real_match_results_path +
                            '/visualize/video_{:0>2d}'.format(video_num))

            for q_num, inputs in enumerate(tqdm.tqdm(data_loader_i)):
                # Initial Pose
                if q_num == 0:
                    cam_obj_poses_before, pose_id_before, _, cam_pose_before \
                        = self.instantiate_render(inputs, renderer_info)
                else:
                    self.prepare_data(inputs)

                    with torch.no_grad():
                        '''extract test image dense kps and descs'''
                        #! TODO
                        if not self.config_test.set_to_eval:
                            for k, m in self.model.net.items():
                                m.train()
                        if self.test_obj_dim == 0:
                            # regrading the inter and intra descriptor as a whole descriptor
                            _, _, desc_real, _, xys_real = self.extract_kps_descpritors(inputs,
                                                                                        multi_scale=config_sim2real[
                                                                                            'ref_img_multi_scale'],
                                                                                        dense_predict=False)
                        else:
                            # Disentangle the descriptor:
                            # The inter descriptor is used to filter the points with similar instance.
                            # The intra descriptor is then used to find the correspondence.
                            # Note that 'repreat_dense_real' designed by R2D2 is not used in our paper.
                            relia_dense_real, repreat_dense_real, desc_intra_real, \
                                desc_inter_real, _ = self.extract_kps_descpritors(inputs,
                                                                                  multi_scale=True, dense_predict=True)

                        '''extract kps, match, calculate pose and update'''
                        if not self.config_test.set_to_eval:
                            for k, m in self.model.net.items():
                                m.eval()

                        for cc in inputs['classes'][1:]:  # 0: background
                            cc_valid_num = torch.where(
                                inputs['label'] == cc)[0].shape[0]
                            # Occulded objects are not involved in the computation.
                            if (cc_valid_num < config_sim2real['valid_num_thr']
                                    and cc != 18) or cc_valid_num < config_sim2real['valid_num_thr']/2:
                                continue

                            '''pose_i-1 in gt'''
                            obj_cc_pose_gt_before = cam_obj_poses_before[np.where(pose_id_before == cc)[
                                0]][0]

                            '''render ref img'''
                            if cc not in self.obj_renderers.keys():
                                self.instantiate_render(
                                    inputs, renderer_info, classes=[cc])
                            obj_cc_pose_img_ref, obj_cc_pose_depth_ref \
                                = self.obj_renderers[cc].render([obj_cc_pose_gt_before])
                            obj_cc_img_ref, obj_cc_mask_ref = preprocess_for_render_img(obj_cc_pose_img_ref,
                                                                                        obj_cc_pose_depth_ref,
                                                                                        self.config_common.net.kwargs.use_depth)
                            inputs_ref = dict(
                                img=obj_cc_img_ref, valid_mask=obj_cc_mask_ref)

                            '''extract sparse kps and descs for certain class of render img'''
                            _, _, desc_ref, _, xys_ref = self.extract_kps_descpritors(inputs_ref,
                                                                                      multi_scale=config_sim2real[
                                                                                          'ref_img_multi_scale'],
                                                                                      dense_predict=False)
                            if self.test_obj_dim > 0:
                                desc_ref = desc_ref[0]
                                xys_ref = xys_ref[0]
                                desc_intra_ref, desc_inter_ref = desc_split(
                                    desc_ref, self.test_obj_dim)
                            else:
                                desc_intra_ref = desc_ref

                            '''when there are few kps in render img'''
                            if desc_intra_ref is None or desc_intra_ref.shape[0] < 1:
                                if cc_valid_num > config_sim2real['valid_num_thr'] \
                                        or (cc_valid_num > config_sim2real['valid_num_thr'] and cc == 18):
                                    count_mma(metric, class_freq, cc, match_kps,
                                              match_dist, np.array([]), mma5s, mma7s)
                                continue
                            else:
                                '''using inter descriptors to filter kps'''
                                if self.test_obj_dim > 0:
                                    desc_cc_center_ref = torch.mean(
                                        desc_inter_ref, dim=0, keepdim=True).t()
                                    xys_new_real, desc_intra_new_real = select_same_obj_pixel(desc_inter_real,
                                                                                              self.test_obj_dim, desc_cc_center_ref,
                                                                                              self.config_evaluate.obj_sim_thr,
                                                                                              relia_dense_real, repreat_dense_real,
                                                                                              desc_intra_real,
                                                                                              self.detector, self.use_relia_score,
                                                                                              self.H, self.W, selectnum=500)
                                else:
                                    xys_new_real = xys_real
                                    desc_intra_new_real = desc_real

                                ''' matching '''
                                '''when there are few kps in real img'''
                                if desc_intra_new_real is None or desc_intra_new_real.shape[0] < 1:
                                    if cc_valid_num > config_sim2real['valid_num_thr'] \
                                            or (cc_valid_num > config_sim2real['valid_num_thr'] and cc == 18):
                                        count_mma(metric, class_freq, cc, match_kps,
                                                  match_dist, np.array([]), mma5s, mma7s)
                                    continue
                                else:
                                    # match with ransac
                                    keypoints_ref, keypoints_real, new_matches = \
                                        meta_matcher(self.config_common.method,
                                                     desc_intra_ref, desc_intra_new_real,
                                                     xys_ref, xys_new_real,
                                                     config_sim2real['ransac_select'],
                                                     self.config_evaluate.positive_sim_thr)
                                    if keypoints_ref is None:
                                        count_mma(metric, class_freq, cc, match_kps,
                                                  match_dist, np.array([]), mma5s, mma7s)
                                        continue

                            '''calculate distance'''
                            inputs_ref['aflow'] = get_aflow(self.camera_intr_44,
                                                            self.camera_intr_inv_44,
                                                            self.bd, self.p3d, cam_pose_before[0],
                                                            inputs['cam_pose'][0], obj_cc_pose_depth_ref,
                                                            iscuda=True).permute(2, 0, 1).unsqueeze(0)
                            inputs_ref['label'] = obj_cc_mask_ref.clone(
                            ).detach()
                            self.get_corr_grid(inputs_ref)

                            match_kp_class, dist, inliers = match_kp_dist(
                                keypoints_ref, keypoints_real,
                                inputs_ref['valid_mask'], inputs_ref['label'],
                                inputs_ref['grid'], self.pix_coords, self.W)
                            if dist.shape[0] == 0:
                                count_mma(metric, class_freq, cc, match_kps,
                                          match_dist, np.array([]), mma5s, mma7s)
                                continue

                            try:
                                new_matches = new_matches[inliers].cpu(
                                ).numpy()
                            except:
                                new_matches = new_matches[inliers]

                            tmp_path = '/visualize/video_{:0>2d}/{:0>4d}_{:0>2d}_match.png'.format(
                                video_num, q_num, cc)
                            # show kps
                            if self.config_test['save_vis_result']:
                                draw_matches(obj_cc_pose_img_ref, inputs['ori_img'][0].cpu().numpy(),
                                             keypoints_ref, keypoints_real,
                                             save_path=self.sim2real_match_results_path+tmp_path)

                            count_mma(metric, class_freq, cc, match_kps,
                                      match_dist, dist, mma5s, mma7s)

                            '''update'''
                            cam_pose_before = inputs['cam_pose']
                            cam_obj_poses_before = inputs['cam_obj_poses']
                            pose_id_before = inputs['pose_id']

            del self.obj_renderers
            del self.cls_ptsxyz_dict
            del self.object_width_dict

        save_kp_performance(match_kps, class_freq, mma5s, mma7s,
                            self.sim2real_match_results_path, self.test_type)

    def match_tracking(self, metric='mean_mma'):
        # some metric and data container
        class_range = self.config_test[self.test_type]['videos']
        match_dist = {i: np.array([]) for i in class_range}
        match_kps = {i: 0 for i in class_range}
        class_freq = {i: 0 for i in class_range}
        mma5s = {i: [] for i in class_range}
        mma7s = {i: [] for i in class_range}

        config_track = self.config_test[self.test_type]

        for video_num, data_loader_vi in self.test_loader.items():
            if not os.path.exists(self.track_results_path+'/visualize/video_{:0>2d}'.format(video_num)):
                os.makedirs(self.track_results_path +
                            '/visualize/video_{:0>2d}'.format(video_num))

            for q_num, inputs in enumerate(tqdm.tqdm(data_loader_vi)):
                if q_num == 0:
                    inputs_0 = copy.deepcopy(inputs)
                    B, _, H, W = inputs_0['img'].shape
                    assert B == 1

                    cc = video_num
                    inputs_0['valid_mask'] = torch.where(
                        inputs_0['label'].cuda() > 0, 1.0, 0.0).unsqueeze(0)
                    inputs_0['valid_mask0_bool'] = torch.where(
                        inputs_0['label'].cuda() > 0, True, False)

                    self.prepare_data(inputs_0)

                    '''extract dense kps and descs of reference image'''
                    with torch.no_grad():
                        _, _, desc_ref, _, xys_ref = self.extract_kps_descpritors(inputs_0,
                                                                                  multi_scale=config_track[
                                                                                      'ref_img_multi_scale'],
                                                                                  dense_predict=False)

                        if self.test_obj_dim > 0:
                            desc_ref = desc_ref[0]
                            xys_ref = xys_ref[0]
                            desc_intra_ref, desc_inter_ref = desc_split(
                                desc_ref, self.test_obj_dim)
                            desc_inter_center_ref = torch.mean(
                                desc_inter_ref, dim=0, keepdim=True).t()
                else:
                    self.prepare_data(inputs)

                    tmp_path = '/visualize/video_{:0>2d}/{:0>4d}_match.png'.format(
                        video_num, q_num)
                    with torch.no_grad():
                        '''extract sparse kps and descs for certain class of current img'''
                        if self.test_obj_dim == 0:
                            _, _, desc_cur, _, xys_cur = self.extract_kps_descpritors(inputs,
                                                                                      multi_scale=config_track[
                                                                                          'ref_img_multi_scale'],
                                                                                      dense_predict=False)

                            xys_new_cur = xys_cur
                            desc_intra_new_cur = desc_cur

                        else:
                            relia_dense_cur, repreat_dense_cur, desc_intra_cur, \
                                desc_inter_cur, _ = self.extract_kps_descpritors(inputs, multi_scale=True,
                                                                                 dense_predict=True)

                            xys_new_cur, desc_intra_new_cur = select_same_obj_pixel(desc_inter_cur,
                                                                                    self.test_obj_dim, desc_inter_center_ref,
                                                                                    self.config_evaluate.obj_sim_thr,
                                                                                    relia_dense_cur, repreat_dense_cur,
                                                                                    desc_intra_cur, self.detector,
                                                                                    self.use_relia_score, H, W, selectnum=500)

                        ''' matching '''
                        '''when there are few kps in test img'''
                        if desc_intra_new_cur is None or desc_intra_new_cur.shape[0] < 1:
                            count_mma(metric, class_freq, cc, match_kps,
                                      match_dist, np.array([]), mma5s, mma7s)
                            continue
                        else:
                            keypoints_ref, keypoints_cur, new_matches = \
                                meta_matcher(self.config_common.method,
                                             desc_intra_ref, desc_intra_new_cur,
                                             xys_ref, xys_new_cur,
                                             config_track['ransac_select'],
                                             self.config_evaluate.positive_sim_thr)

                            '''calculate distance'''
                            dist = cal_dist(keypoints_ref, keypoints_cur,
                                            inputs_0['depth'].numpy(
                                            ), inputs['K'],
                                            inputs_0['cam_obj_poses'], inputs['cam_obj_poses'])
                            count_mma(metric, class_freq, cc, match_kps,
                                      match_dist, dist, mma5s, mma7s)

                            if self.config_test['save_vis_result']:
                                # draw_matches(inputs_0['ori_img'][0].cpu().numpy(), inputs['ori_img'][0].cpu().numpy(),
                                #              keypoints_ref, keypoints_cur)
                                draw_matches_v2(inputs_0['ori_img'][0].cpu().numpy(), xys_ref.cpu().numpy(),
                                                inputs['ori_img'][0].cpu().numpy(
                                ), xys_new_cur.cpu().numpy(),
                                    good_matches=new_matches,
                                    save_path=self.track_results_path+tmp_path)

        save_kp_performance(match_kps, class_freq, mma5s, mma7s,
                            self.track_results_path, self.test_type)

    def sim2real_pose_estimation(self,):
        # some metric and data container
        results = {}
        cls_ptsxyz_dict = {}
        for c in range(1, self.config_common.public_params.num_class):
            cls_ptsxyz_dict[c] = np.array([])

        config_pose = self.config_test[self.test_type]

        for video_num, data_loader_vi in self.test_loader.items():
            obj_pred_poses = {c: [] for c in range(
                1, self.config_common.public_params.num_class)}

            for q_num, inputs in enumerate(tqdm.tqdm(data_loader_vi)):
                '''prepare real image '''
                self.prepare_data(inputs)
                results['keyfram_{:0>4d}'.format(q_num)] = {}

                #! real image inference
                with torch.no_grad():
                    if not self.config_test.set_to_eval:
                        for k, m in self.model.net.items():
                            m.train()
                    relia_dense_real, repreat_dense_real, desc_intra_real, \
                        desc_inter_real, xys_real = self.extract_kps_descpritors(inputs,
                                                                                 multi_scale=True,
                                                                                 dense_predict=True)

                    for cc in inputs['classes'][1:]:
                        #! template image inference
                        temp_kps_desc_cc = self.template_data_inference(cc)

                        #! matching
                        indexes_temp, matches, kps_temp, kps_real = one2all_match(relia_dense_real,
                                                                                  repreat_dense_real, desc_intra_real,
                                                                                  desc_inter_real, xys_real,
                                                                                  temp_kps_desc_cc['kps'],
                                                                                  temp_kps_desc_cc['descs'],
                                                                                  self.config_common.method, self.test_obj_dim,
                                                                                  config_pose['use_gt_box'],
                                                                                  self.detector, self.use_relia_score,
                                                                                  self.config_evaluate.obj_sim_thr,
                                                                                  self.config_evaluate.positive_sim_thr,
                                                                                  config_pose['template_top_num'])

                        #! select top num! select best! (from high -- low)
                        inliers_list = []
                        obj_cc_pose_pred_1_list = []
                        for i in range(0, config_pose['template_top_num']):
                            temp_inputs = self.template_dataset[cc][indexes_temp[i]]
                            try:
                                if torch.is_tensor(kps_real[i]):
                                    # num, 2
                                    kp_real = kps_real[i][matches[i][:, 0]].cpu(
                                    ).numpy()
                                else:
                                    kp_real = kps_real[i][matches[i]
                                                          [:, 0].cpu().numpy()]
                            except:
                                break
                            if kp_real.shape[0] >= 4:
                                if torch.is_tensor(kps_temp[i]):
                                    kp_temp = kps_temp[i][matches[i][:, 1]].float(
                                    ).cpu().numpy()  # num, 3
                                else:
                                    kp_temp = kps_temp[i][matches[i]
                                                          [:, 1].cpu().numpy()]

                                # draw_keypoints(temp_inputs['ori_img'], inputs['ori_img'][0].cpu().numpy(),
                                #                 kp_temp, kp_real)

                                kp_obj_3d = img_coord_2_obj_coord(kp_temp,
                                                                  temp_inputs['depth'],
                                                                  temp_inputs['K'],
                                                                  temp_inputs['cam_obj_poses'])
                                obj_cc_pose_pred_1, inliers = pnp(kp_obj_3d[:, :3],
                                                                  kp_real,
                                                                  inputs['K'])
                                if inliers is None:
                                    continue
                                inliers_list.append(inliers.shape[0])
                                obj_cc_pose_pred_1_list.append(
                                    obj_cc_pose_pred_1)
                            else:
                                break

                        '''pose_i in gt'''
                        obj_cc_pose_gt_1 = inputs['cam_obj_poses'][np.where(
                            inputs['pose_id'] == cc)[0]][0]
                        if cls_ptsxyz_dict.get(cc, None).shape[0] == 0:
                            cls_ptsxyz_dict[cc] = np.loadtxt(os.path.join(config_pose['ref_model_root'],
                                                                          '{}/points.xyz'.format(cc)), dtype=np.float32)

                        if len(inliers_list) > 0:
                            max_inliers_index = inliers_list.index(
                                max(inliers_list))
                            obj_cc_pose_pred_1 = obj_cc_pose_pred_1_list[max_inliers_index]
                            obj_pred_poses[cc].append(obj_cc_pose_pred_1)
                        else:
                            log_add(pred_pose=np.eye(4), gt_pose=obj_cc_pose_gt_1, pts3d=cls_ptsxyz_dict[cc],
                                    obj_add_result=results['keyfram_{:0>4d}'.format(q_num)], obj_class=cc)
                            continue

                        # #! log
                        # tmp_path = '{:0>4d}_class_{:0>2d}_kp.png'.format(q_num, cc)
                        # save_path_query_cc_mostconf_kp = os.path.join(self.pose_results_path,
                        #                                             '{}'.format(video_num), tmp_path)

                        #! pose result
                        '''calculate add and save result'''
                        log_add(pred_pose=obj_cc_pose_pred_1, gt_pose=obj_cc_pose_gt_1,
                                pts3d=cls_ptsxyz_dict[cc],
                                obj_add_result=results['keyfram_{:0>4d}'.format(
                                    q_num)],
                                obj_class=cc)

            store_json(results, os.path.join(self.pose_results_path,
                       '{}_result.json'.format(video_num)))
            results = {}

        self.combine_all_obj_add()
