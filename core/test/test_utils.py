import numpy as np
import os
from functools import partial

import torch
import torchvision.transforms as tvf
import trimesh
import pyrender
import ipdb
import cv2

import torch.nn as nn
from core.tools.transforms_tools import img_coord_2_obj_coord, compute_bbox
from core.tools.metric import cal_adds_dis, cal_add_dis
from core.utils import store_json


RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose(
    [tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


def count_mma(metric, class_freq, cc, match_kps, match_dist, dist, mma5s, mma7s):
    class_freq[cc] += 1
    match_kps[cc] += dist.shape[0]

    if metric == 'total_mma':
        if dist.shape[0] > 0:
            match_dist[cc] = np.append(match_dist[cc], dist)  # np.array
            print('mma-5:', np.mean(dist < 5))
        else:
            print('mma-5: 0')
    else:
        if dist.shape[0] > 4:
            match_dist[cc] = np.append(
                match_dist[cc], np.array(dist.mean()))  # np.array
            mma5s[cc].append(np.mean(dist < 5))
            mma7s[cc].append(np.mean(dist < 7))
            print('mma-5:', np.mean(dist < 5))
        else:
            match_dist[cc] = np.append(match_dist[cc], np.array(100))
            mma5s[cc].append(0)
            mma7s[cc].append(0)
            print('mma-5: 0')


def cal_dist(kps_scene, kps_ref, depth1, camera_intr_44, obj_pose1, obj_pose2):
    keypoints_1_obj = img_coord_2_obj_coord(kps_scene, depth1[0],
                                            camera_intr_44, obj_pose1)  # Num, 3
    keypoints_ref_gt = (camera_intr_44 @ obj_pose2 @ keypoints_1_obj.T).T
    keypoints_ref_gt = keypoints_ref_gt[:, :3]
    keypoints_ref_gt = keypoints_ref_gt[:, :2] / keypoints_ref_gt[:, 2:]

    dist = np.linalg.norm((keypoints_ref_gt - kps_ref), axis=1)

    return dist


def log_add(pred_pose, gt_pose, pts3d, obj_add_result, obj_class):
    cc_add = cal_add_dis(pred_pose[:3], gt_pose[:3], pts3d)
    #cc_adds = cal_adds_dis(pred_pose[:3], gt_pose[:3], pts3d)
    cc_adds = cal_adds_dis(pred_pose[:3], gt_pose[:3], pts3d)

    print('cc_add:', cc_add)
    print('cc_adds:', cc_adds)
    # obj_add_result[(obj_class, 'add')].append(cc_add)
    # obj_add_result[(obj_class, 'adds')].append(cc_adds)
    # np.savetxt(save_path_add, np.array([cc_add, cc_adds]))
    obj_add_result[str(obj_class)] = ['{:0>4f}'.format(
        cc_add), '{:0>4f}'.format(cc_adds)]


def get_renderer_info(ref_model_path, classes):
    renderer_data = {}

    for i in classes:
        obj_i_path = os.path.join(ref_model_path, str(i))

        renderer_data[(i, 'obj_path')] = os.path.join(
            obj_i_path, 'textured.obj')

        i_corners_path = os.path.join(obj_i_path, 'corners.txt')
        renderer_data[(i, 'corners')] = np.loadtxt(
            i_corners_path)  # for pose visulization

        renderer_data[(i, 'pts_path')] = os.path.join(obj_i_path, 'points.xyz')

        renderer_data['resolution'] = (480, 640)

    return renderer_data


class Renderer:
    def __init__(self, model_paths, cam_K, H, W, scale=None):
        if not isinstance(model_paths, list):
            print("model_paths have to be list")
            raise RuntimeError
        self.scene = pyrender.Scene(
            ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
        self.camera = pyrender.IntrinsicsCamera(
            fx=cam_K[0, 0], fy=cam_K[1, 1], cx=cam_K[0, 2], cy=cam_K[1, 2], znear=0.1, zfar=2.0)
        self.cam_node = self.scene.add(self.camera, pose=np.eye(4))
        self.mesh_nodes = []

        for model_path in model_paths:
            print('model_path', model_path)
            obj_mesh = trimesh.load(model_path)
            if abs((scale-1)) > 1e-4:
                obj_mesh.vertices = obj_mesh.vertices*scale
                obj_mesh.faces = obj_mesh.faces*scale
            obj_mesh.vertices = obj_mesh.vertices
            obj_mesh.faces = obj_mesh.faces
            #colorVisual = obj_mesh.visual.to_color()
            mesh = pyrender.Mesh.from_trimesh(obj_mesh)
            mesh_node = self.scene.add(mesh, pose=np.eye(
                4), parent_node=self.cam_node)  # Object pose parent is cam
            self.mesh_nodes.append(mesh_node)

        self.H = H
        self.W = W

        self.r = pyrender.OffscreenRenderer(self.W, self.H)
        self.glcam_in_cvcam = np.array([[1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]])
        self.cvcam_in_glcam = np.linalg.inv(self.glcam_in_cvcam)

    def render(self, ob_in_cvcams):
        assert isinstance(ob_in_cvcams, list)
        for i, ob_in_cvcam in enumerate(ob_in_cvcams):
            ob_in_glcam = self.cvcam_in_glcam.dot(ob_in_cvcam)
            self.scene.set_pose(self.mesh_nodes[i], ob_in_glcam)
        color, depth = self.r.render(self.scene)  # depth: float
        return color, depth


def render_window(obj_renderer, ob2cam, K, object_width, image_size):
    '''
    @ob2cam: 4x4 mat ob in opencv cam
    '''

    bbox = compute_bbox(ob2cam, K, object_width, scale=(1000, 1000, 1000))
    rgb, depth = obj_renderer.render([ob2cam])
    #depth = (depth*1000).astype(np.uint16)
    #show_depth_image(rgb, rgb, depth, depth)
    # ipdb.set_trace()
    #render_rgb, render_depth = crop_bbox(rgb, depth, bbox, image_size)
    return rgb, depth, bbox[:, [1, 0]]


def preprocess_for_render_img(img, depth, use_depth=True, depth_max_min_normalize=False, max_depth=2.0):
    label = torch.from_numpy(np.where(depth > 0, 1.0, 0.0)).unsqueeze(
        0).unsqueeze(0).cuda()
    if use_depth:
        depth = tvf.ToTensor()(np.float32(depth))
        if depth_max_min_normalize:
            depth = (depth - depth.min())/(depth.max()-depth.min())
        else:
            depth = depth / max_depth
        depth = (depth - 0.45) / 0.225
        return torch.cat((norm_RGB(img.copy()), depth.repeat(3, 1, 1)), 0).cuda().unsqueeze(0), label
    else:
        return norm_RGB(img.copy()).cuda().unsqueeze(0), label


def save_kp_performance(match_kps, class_freq, mma5s, mma7s, track_results_path, test_type):
    mma5_save = [sum(v)/(class_freq[key]+1e-5) for key, v in mma5s.items()]
    mma7_save = [sum(v)/(class_freq[key]+1e-5) for key, v in mma7s.items()]
    kps_save = [v/(class_freq[key]+1e-5) for key, v in match_kps.items()]
    np.savetxt(os.path.join(track_results_path,
               test_type+'_mma5.txt'), np.array(mma5_save))
    np.savetxt(os.path.join(track_results_path,
               test_type+'_mma7.txt'), np.array(mma7_save))
    np.savetxt(os.path.join(track_results_path,
               test_type+'_kps.txt'), np.array(kps_save))

    mean_info = {}
    if test_type == 'match_tracking' or test_type == 'sim2real_match':
        mean_info['kps_mean'] = np.array(kps_save).mean()
        mean_info['mma5_mean'] = np.array(mma5_save).mean()
        mean_info['mma7_mean'] = np.array(mma7_save).mean()

    store_json(mean_info, os.path.join(
        track_results_path, test_type+'_mean_info.json'))



def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    # _, R_exp, t = cv2.solvePnP(points_3d,
    #                            points_2d,
    #                            camera_matrix,
    #                            dist_coeffs,
    #                            flags=method)
    # , None, None, False, cv2.SOLVEPNP_UPNP)
    try:
        np.random.seed(0)
        _, R_exp, t, inliers = cv2.solvePnPRansac(points_3d,
                                                  points_2d,
                                                  camera_matrix,
                                                  dist_coeffs,
                                                  reprojectionError=6.0,
                                                  iterationsCount=5000, flags=cv2.SOLVEPNP_EPNP)
        #cv2.SOLVEPNP_EPNP , flags=cv2.SOLVEPNP_EPNP
    except:
        _, R_exp, t = cv2.solvePnP(points_3d,
                                   points_2d,
                                   camera_matrix,
                                   dist_coeffs,
                                   flags=method)
        inliers = np.expand_dims(np.arange(points_3d.shape[0]), 1)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    r_t = np.concatenate([R, t], axis=-1)
    return np.concatenate((r_t, [[0, 0, 0, 1]]), axis=0), inliers


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        # pix_coords[..., 0] /= self.width - 1
        # pix_coords[..., 1] /= self.height - 1
        # pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def get_aflow(cam_intr, cam_intr_inv, back_depth, proj3d, p1, p2, d1, iscuda=False):
    T21 = torch.tensor(np.dot(np.linalg.inv(p2), p1), dtype=torch.float32).unsqueeze(0)#T2w*Tw1
    cam_points = back_depth(torch.tensor(d1).unsqueeze(0), cam_intr_inv)
    pix_coords = proj3d(cam_points, cam_intr, T21)
    if iscuda:
        return pix_coords[0]
    else:
        return pix_coords[0].numpy()
