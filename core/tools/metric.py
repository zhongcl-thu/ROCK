import ipdb
import numpy as np
from scipy import spatial


def cal_adds_dis(pred_pose, gt_pose, cls_ptsxyz):
    pred_pts = np.dot(cls_ptsxyz, pred_pose[:, :3].T) + pred_pose[:, 3]
    gt_pts = np.dot(cls_ptsxyz, gt_pose[:, :3].T) + gt_pose[:, 3]

    nn_index = spatial.cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1)

    e = nn_dists.mean()
    return e


def cal_add_dis(pred_pose, gt_pose, cls_ptsxyz):
    pred_pts = np.dot(cls_ptsxyz, pred_pose[:, :3].T) + pred_pose[:, 3]
    gt_pts = np.dot(cls_ptsxyz, gt_pose[:, :3].T) + gt_pose[:, 3]
    mean_dist = np.mean(np.linalg.norm(pred_pts - gt_pts, axis=-1))
    return mean_dist


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def cal_auc(add_dis, max_dis=0.1):
    D = np.array(add_dis)
    D[np.where(np.isnan(D) == True)[0]] = np.inf
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100
