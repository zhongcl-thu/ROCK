import cv2

import numpy as np

import ipdb
import heapq

import torch
import torch.nn.functional as F
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from core.tools.transforms_tools import desc_split


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, base_thr=0.7, refine_thr=0.7):
        torch.nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)
        self.base_thr = base_thr
        self.refine_thr = refine_thr

    def forward(self, base, refine):
        assert base.dim() == refine.dim() == 4

        # local maxima
        maxima = (base == self.max_filter(base))
        # remove low peaks
        maxima *= (base >= self.base_thr)
        maxima *= (refine >= self.refine_thr)

        aa = torch.where(maxima[0, 0] == True)

        return aa


def select_same_obj_pixel(desc_inter_real, inter_obj_dim, desc_cc_center_ref, obj_sim_thr,
                          relia_dense_real, repreat_dense_real, desc_intra_real,
                          detector, use_relia_score, H, W, selectnum=500):
    desc_inter_one_dim_real = desc_inter_real[0].reshape(inter_obj_dim, -1).t()
    desc_inter_sim_real = torch.mm(desc_inter_one_dim_real, desc_cc_center_ref)

    valid_desc_mask_real = (desc_inter_sim_real >
                            obj_sim_thr).reshape(-1, H, W)

    relia_dense_mask_real = valid_desc_mask_real * relia_dense_real.squeeze(0)
    repreat_dense_mask_real = valid_desc_mask_real * \
        repreat_dense_real.squeeze(0)

    xys_new_ref, desc_intra_new_ref, _ = dense_selection(relia_dense_mask_real,
                                                         repreat_dense_mask_real, desc_intra_real.squeeze(
                                                             0),
                                                         detector, selectnum=selectnum,
                                                         use_relia_score=use_relia_score)

    return xys_new_ref, desc_intra_new_ref


def dense_selection(reliability, repeatability, descriptors, detector, selectnum=2000, bbox=None,
                    use_relia_score=False):
    #XY, S, D = [], [], []
    if use_relia_score:
        y, x = detector(base=reliability.unsqueeze(0),
                        refine=repeatability.unsqueeze(0))
    else:
        y, x = detector(base=repeatability.unsqueeze(0),
                        refine=reliability.unsqueeze(0))  # nms
    c_0 = reliability[0, y, x]  # reliability
    q_0 = repeatability[0, y, x]
    d_0 = descriptors[:, y, x].t()

    score_0 = c_0 * q_0  # scores = reliability * repeatability
    XY_0 = torch.stack([x, y], dim=-1)

    if score_0.shape[0] > selectnum:
        idxs = torch.argsort(score_0)[-selectnum:]
        XY_0 = XY_0[idxs]
        d_0 = d_0[idxs]
        score_0 = score_0[idxs]

    if bbox is not None:
        XY_0 += torch.tensor([bbox[0, 0], bbox[0, 1]]).cuda()

    return XY_0, d_0, score_0


def meta_matcher(kp_type, desc_1, desc_2, xys_1, xys_2,
                 ransac_select, positive_sim_thr):
    if kp_type != 'SIFT' and kp_type != 'SURF':
        matches = mnn_matcher(desc_1, desc_2)
        '''remove some unsimilar points '''
        dess_1, dess_2 = desc_1[matches[:, 0]], desc_2[matches[:, 1]]
        valid_pos = torch.where(
            torch.sum(dess_1*dess_2, dim=1) > positive_sim_thr)
        new_matches = matches[valid_pos]
    else:
        new_matches = match_descriptors(desc_1, desc_2)

    if torch.is_tensor(xys_2):
        kps_1 = xys_1[new_matches[:, 0]][:, :2].cpu().numpy()  # num, 2
        kps_2 = xys_2[new_matches[:, 1]][:, :2].cpu().numpy()  # num, 3
    else:
        kps_1, kps_2 = [], []
        for index in new_matches:
            kps_1.append([xys_1[index[0]].pt[0], xys_1[index[0]].pt[1]])
            kps_2.append([xys_2[index[1]].pt[0], xys_2[index[1]].pt[1]])
        kps_1 = np.concatenate(kps_1).reshape(-1, 2)
        kps_2 = np.concatenate(kps_2).reshape(-1, 2)

    ''' ransac  '''
    if ransac_select and new_matches.shape[0] > 4:
        np.random.seed(0)
        model, inliers = ransac(
            (kps_1, kps_2),
            ProjectiveTransform, min_samples=4,
            residual_threshold=4, max_trials=1000)

        if inliers is None:
            new_matches = kps_1 = kps_2 = None
        else:
            new_matches = new_matches[inliers]
            kps_1 = kps_1[inliers]
            kps_2 = kps_2[inliers]

    return kps_1, kps_2, new_matches


def mnn_matcher(descriptors_a, descriptors_b):
    # descriptors shape: (num, 128)

    if descriptors_a.dim() == 3:
        raise ValueError('descriptors must have 2 dims')

    elif descriptors_a.dim() == 2:
        # shape (des_a.shape[0], des_a.shape[0])
        sim = descriptors_a @ descriptors_b.t()
        try:
            nn12 = torch.max(sim, dim=1)[1]
        except:
            ipdb.set_trace()
        nn21 = torch.max(sim, dim=0)[1]
        ids1 = torch.arange(0, sim.shape[0], device=descriptors_a.device)
        mask = (ids1 == nn21[nn12])
        matches = torch.stack([ids1[mask], nn12[mask]])
    else:
        raise ValueError('descriptors must have 2 or 3 dims')

    return matches.t()


def one2all_match(relia_dense_1, repreat_dense_1,
                  desc_intra_1, desc_inter_1,
                  xys_1, xy_lists_2, desc_lists_2,
                  method, inter_obj_dim, use_gt_box,
                  detector, use_relia_score,
                  obj_sim_thr, positive_sim_thr, template_top_num=1):

    matches_num_2 = []
    matches_2 = []
    matches_xys_1 = []

    if inter_obj_dim > 0:
        _, _, H, W = relia_dense_1.shape

    for j in range(len(xy_lists_2)):
        ''' split description'''
        if inter_obj_dim > 0:
            desc_intra_j_2, desc_inter_j_2 = desc_split(
                desc_lists_2[j], inter_obj_dim)
        else:
            desc_intra_j_2 = desc_lists_2[j]

        if desc_intra_j_2.shape[0] > 0:
            ''' obj'''
            if inter_obj_dim > 0:
                if use_gt_box:
                    raise ValueError('todo')
                else:
                    desc_inter_center_2 = torch.mean(
                        desc_inter_j_2, dim=0, keepdim=True).t()
                    desc_inter_one_dim_1 = desc_inter_1[0].reshape(
                        inter_obj_dim, -1).t()
                    desc_inter_sim_2_1 = torch.mm(
                        desc_inter_one_dim_1, desc_inter_center_2)
                    valid_obj_des = (desc_inter_sim_2_1 >
                                     obj_sim_thr).reshape(-1, H, W)
                if method == 'ours':
                    relia_dense_obj_mask_1 = valid_obj_des * \
                        relia_dense_1.squeeze(0)
                    repreat_dense_obj_mask_1 = valid_obj_des * \
                        repreat_dense_1.squeeze(0)
                    xys_new_1, desc_intra_new_1, _ = dense_selection(relia_dense_obj_mask_1,
                                                                     repreat_dense_obj_mask_1,
                                                                     desc_intra_1.squeeze(
                                                                         0),
                                                                     detector, selectnum=500,
                                                                     use_relia_score=use_relia_score)
                else:
                    raise ValueError('unknown method')

            else:
                desc_intra_new_1 = desc_intra_1
                xys_new_1 = xys_1

            ''' matching '''
            if desc_intra_new_1.shape[0] != 0:
                matches = mnn_matcher(desc_intra_new_1, desc_intra_j_2)

                '''remove some unsimilar points '''
                dess1, dess2 = desc_intra_new_1[matches[:, 0]
                                                ], desc_intra_j_2[matches[:, 1]]
                valid_pos = torch.where(
                    torch.sum(dess1*dess2, dim=1) > positive_sim_thr)
                new_matches = matches[valid_pos]

                matches_2.append(new_matches)
                matches_num_2.append(new_matches.shape[0])
                matches_xys_1.append(xys_new_1)

            else:
                matches_num_2.append(0)
                matches_2.append(np.empty([1, 1]))
                matches_xys_1.append(np.empty(0))
        else:
            matches_num_2.append(0)
            matches_2.append(np.empty([1, 1]))
            matches_xys_1.append(np.empty(0))

    '''find the most confident view'''
    index = heapq.nlargest(template_top_num, range(len(matches_num_2)),
                           matches_num_2.__getitem__)

    matches_2 = np.array(matches_2)[index]
    xy_lists_2 = np.array(xy_lists_2)[index]
    matches_xys_1 = np.array(matches_xys_1)[index]

    return index, matches_2, xy_lists_2, matches_xys_1


def match_kp_dist(pos_1_xy, pos_2_xy, valid_mask, label, grid, pix_coords, W):
    pos_1 = np.array(np.round(pos_1_xy[:, 1])
                     * W + pos_1_xy[:, 0], dtype=np.int)
    v_m = valid_mask.reshape(1, -1).cpu().numpy()
    label = label.reshape(1, -1).cpu().numpy()
    label_bool = np.where(label > 0, True, False)
    label_bool[~v_m] = False
    try:
        v_m_i = label_bool[0][pos_1]
    except:
        ipdb.set_trace()

    if grid == None:
        match_kp_class = None
        dist = None
    else:
        try:
            pix_12 = F.grid_sample(pix_coords, grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)  # (B,2,H,W)
        except:
            pix_coords = pix_coords.to(valid_mask.device)
            pix_12 = F.grid_sample(pix_coords, grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)  # (B,2,H,W)

        gt_pos_2 = torch.stack((pix_12[0][1].reshape(-1)[pos_1],
                                pix_12[0][0].reshape(-1)[pos_1]), 1).cpu().numpy()
        pos_2_xy = np.stack((pos_2_xy[:, 0], pos_2_xy[:, 1]), 1)
        pos_1_xy = np.stack((pos_1_xy[:, 0], pos_1_xy[:, 1]), 1)
        dist = np.linalg.norm((pos_2_xy - gt_pos_2), axis=1)[v_m_i]
        match_kp_class = label[0][pos_1[v_m_i]]

    return match_kp_class, dist, v_m_i


def extract_kps_descs_handcrafted(img, kp_type):
    img_numpy_copy = img.copy()
    if kp_type == 'SIFT':
        sift = cv2.xfeatures2d_SIFT.create()
        xys, desc_distinct = sift.detectAndCompute(img_numpy_copy, None)
    elif kp_type == 'SURF':
        surf = cv2.xfeatures2d.SURF_create(400)
        xys, desc_distinct = surf.detectAndCompute(img_numpy_copy, None)

    return xys, desc_distinct


def extract_single_scale(net, img, detector=None, kp_type='ours', un=True, select=True,
                         selectnum=2000, valid_mask=None, use_relia_score=False, select_xy=None):

    B, _, _, _ = img.shape
    XY, S, D = [], [], []

    with torch.no_grad():
        if isinstance(net, dict):
            features = net["encoder"](imgs=[img])
            res = net["decoder"](features, render_mask=None)
        else:
            res = net(input_feas_list=[img])

    if valid_mask is None:
        valid_mask = torch.ones_like(res['reliability'][0])

    # get output and reliability map
    descriptors = res['descriptors'][0]
    if un:
        if kp_type == 'ours' or kp_type == 'don':
            res['reliability'][0] = torch.exp(-res['reliability'][0])
    else:
        res['reliability'] = [torch.ones_like(res['repeatability'][0])]

    # ipdb.set_trace()
    # show_pair_responses(ori_img, ori_img, res['reliability'][0][0, 0], res['reliability'][0][0, 0])
    reliability = res['reliability'][0] * valid_mask
    if use_relia_score:
        repeatability = torch.ones_like(reliability)
    else:
        repeatability = res['repeatability'][0] * valid_mask

    if select:
        if select_xy is None:
            for i in range(0, B):
                if use_relia_score:
                    y_i, x_i = detector(base=reliability[i].unsqueeze(0),
                                        refine=repeatability[i].unsqueeze(0))  # nms
                else:
                    y_i, x_i = detector(base=repeatability[i].unsqueeze(0),
                                        refine=reliability[i].unsqueeze(0))  # nms

                c_i = reliability[i, 0, y_i, x_i]  # reliability
                q_i = repeatability[i, 0, y_i, x_i]
                d_i = descriptors[i, :, y_i, x_i].t()

                score_i = c_i * q_i  # scores = reliability * repeatability
                XY_i = torch.stack([x_i, y_i], dim=-1)

                if score_i.shape[0] > selectnum:
                    idxs = torch.argsort(score_i)[-selectnum:]
                    XY_i = XY_i[idxs]
                    d_i = d_i[idxs]
                    score_i = score_i[idxs]

                XY.append(XY_i)
                S.append(score_i)
                D.append(d_i)

            return XY, D, S
        else:
            valid_mask = torch.where(valid_mask > 0, True, False)
            for i in range(0, B):
                valid_mask_select = valid_mask[i, 0,
                                               select_xy[i][:, 1], select_xy[i][:, 0]]
                select_xy_i = select_xy[i][valid_mask_select.cpu().numpy()]
                d_i = descriptors[i, :, select_xy_i[:, 1],
                                  select_xy_i[:, 0]].t()
                XY.append(select_xy_i)
                D.append(d_i)

            return XY, D, _

    else:
        return reliability, repeatability, descriptors


def extract_multiscale(net, img, detector, kp_type, un, scale_f=2**0.25,
                       min_scale=0.0, max_scale=1, min_size=256, max_size=1024,
                       verbose=False, valid_mask=None, selectnum=2000, ori_img=None,
                       cc=None, use_relia_score=False, save_path=None):

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1, "should be a batch with a single RGB image"  # and three == 3

    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s+0.001 >= max(min_scale, min_size / max(H, W)):
        if s-0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            # if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors

            with torch.no_grad():
                if isinstance(net, dict):
                    features = net["encoder"](imgs=[img])
                    res = net["decoder"](features, render_mask=None)
                else:
                    res = net(input_feas_list=[img])

            if valid_mask is None:
                valid_mask = torch.ones_like(res['repeatability'][0])
            else:
                if valid_mask.dtype is torch.bool:
                    valid_mask = torch.where(valid_mask == True, 1.0, 0.0)
                valid_mask = F.interpolate(
                    valid_mask, (nh, nw), mode='nearest')

            # get output and reliability map
            descriptors = res['descriptors'][0]
            if un:
                if kp_type == 'ours' or kp_type == 'don':
                    res['reliability'][0] = torch.exp(-res['reliability'][0])
            else:
                res['reliability'] = [torch.ones_like(res['repeatability'][0])]

            # if s > 0.5:
            #     ipdb.set_trace()
            #     show_pair_responses(ori_img, ori_img, res['reliability'][0][0, 0], res['reliability'][0][0, 0], save_path)
            reliability = res['reliability'][0] * valid_mask
            repeatability = res['repeatability'][0] * valid_mask

            # normalize the reliability for nms
            # extract maxima and descs
            if use_relia_score:
                repeatability = torch.ones_like(reliability)
                # save_path='response_{}.png'.format(cc))
                y, x = detector(base=reliability, refine=repeatability)
            else:
                y, x = detector(base=repeatability, refine=reliability)

            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)

        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        #img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
        img = F.interpolate(img, (nh, nw), mode='nearest')

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    # scores = reliability * repeatability
    scores = torch.cat(C) * torch.cat(Q)
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)

    '''select'''
    idxs = torch.argsort(scores)[-selectnum:]
    XYS = XYS[idxs]
    D = D[idxs]
    scores = scores[idxs]

    return XYS, D, scores
