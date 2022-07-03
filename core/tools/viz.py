import ipdb
import numpy as np
import matplotlib.pyplot as pl
import cv2

import torch
import torch.nn.functional as F

from core.nets.sampler import FullSampler


def show_flow_mask_image(I1, I2, mask1, mask2, aflow, save=False):
    '''
    #img_a: np H,W,C 
    #img_b: np H W C
    #mask: np H W
    #aflow: 2 H W
    '''
    if torch.is_tensor(I1):
        I1 = I1.cpu().numpy()
    if torch.is_tensor(I2):
        I2 = I2.cpu().numpy()
    if torch.is_tensor(mask1):
        mask1 = mask1.cpu().numpy()
    if torch.is_tensor(aflow):
        pass
    else:
        aflow = torch.from_numpy(aflow)

    pl.figure()
    pl.subplot(221)
    pl.imshow(I1)
    pl.subplot(222)
    pl.imshow(I2)
    pl.subplot(223)
    pl.imshow(mask1)

    pl.subplot(224)

    grid = FullSampler._aflow_to_grid(
        aflow.unsqueeze(0), I2.shape[0], I2.shape[1])
    # border_mask = np.where(grid.abs() <= 1, True, False)[0]
    # border_mask = border_mask[:, :, 0] * border_mask[:, :, 1]
    # mask = (mask>0) * border_mask

    I2 = I2.transpose(2, 0, 1)

    I12 = F.grid_sample(torch.from_numpy(I2).unsqueeze(0).float(), grid,
                        mode='bilinear', padding_mode='zeros', align_corners=True)

    mask12 = F.grid_sample(torch.from_numpy(mask2).unsqueeze(0).unsqueeze(0).float(), grid,
                           mode='nearest', padding_mode='zeros', align_corners=True)

    valid_pixel = mask12[0, 0].cpu().numpy().astype(np.int32) == mask1
    mask = valid_pixel * np.where(mask1 > 0, True, False)  # * border_mask
    I12 *= mask
    I12 = I12.numpy()[0].transpose(1, 2, 0)

    pl.imshow(I12.astype(np.uint8))
    #pl.imshow(mask12[0, 0].numpy().astype(np.uint8))
    if save:
        pl.savefig('test2.png')
    else:
        pl.show()
    pl.close()


def show_depth_image(I1, I2, depth1, depth2, save=False, save_name='depth.png'):
    '''
    #img_a: np H,W,C 
    #img_b: np H W C
    #mask: np H W
    #aflow: 2 H W
    '''
    if torch.is_tensor(I1):
        I1 = I1.cpu().numpy()
    if torch.is_tensor(I2):
        I2 = I2.cpu().numpy()
    if torch.is_tensor(depth1):
        depth1 = depth1.cpu().numpy()
    if torch.is_tensor(depth2):
        depth2 = depth2.cpu().numpy()

    if I2 is None:
        pl.figure()
        pl.subplot(121)
        pl.imshow(I1)
        pl.subplot(122)
        pl.imshow(depth1, cmap='magma')

    else:
        pl.figure()
        pl.subplot(221)
        pl.imshow(I1)
        pl.subplot(222)
        pl.imshow(I2)

        pl.subplot(223)
        pl.imshow(depth1, cmap='magma')
        pl.subplot(224)
        pl.imshow(depth2, cmap='magma')

    if save:
        pl.savefig(save_name)
    else:
        pl.show()
    pl.close()


def show_mask_image(I1, I2, mask1, mask2, save=False):
    '''
    #img_a: np H,W,C 
    #img_b: np H W C
    #mask: np H W
    #aflow: 2 H W
    '''
    if torch.is_tensor(I1):
        I1 = I1.cpu().numpy()
    if torch.is_tensor(I2):
        I2 = I2.cpu().numpy()
    if torch.is_tensor(mask1):
        mask1 = mask1.cpu().numpy()
    if torch.is_tensor(mask2):
        mask2 = mask2.cpu().numpy()

    if I1.shape[2] != 3:
        I1 = I1.transpose(1, 2, 0)
    if I2.shape[2] != 3:
        I2 = I2.transpose(1, 2, 0)

    pl.figure()
    pl.subplot(221)
    pl.imshow(I1)
    pl.subplot(222)
    pl.imshow(I2)
    pl.subplot(223)
    pl.imshow(mask1)
    pl.subplot(224)
    pl.imshow(mask2)

    if save:
        pl.savefig('test1.png')
    else:
        pl.show()
    pl.close()


def draw_keypoints(image1, image2, keypoints1, keypoints2, save_path=None):
    # image1/2  torch.tensor
    # keypoints1/2 torch.tensor
    if torch.is_tensor(image1):
        image1 = image1.cpu().numpy().copy()
    else:
        image1 = image1.copy()
    if torch.is_tensor(image2):
        image2 = image2.cpu().numpy().copy()
    else:
        image2 = image2.copy()
    if torch.is_tensor(keypoints1):
        keypoints1 = keypoints1.cpu().numpy().astype(int)
    if torch.is_tensor(keypoints2):
        keypoints2 = keypoints2.cpu().numpy().astype(int)

    keypoints1 = keypoints1.astype(int)
    keypoints2 = keypoints2.astype(int)

    for center in keypoints1:
        cv2.circle(image1, (center[0], center[1]), 4, (0, 255, 0), 3, 0)

    if keypoints2 is not None:
        for center in keypoints2:
            cv2.circle(image2, (center[0], center[1]), 4, (0, 255, 0), 3, 0)

        pl.subplot(1, 2, 1)
        pl.imshow(image1)

        pl.subplot(1, 2, 2)
        pl.imshow(image2)
    else:
        pl.imshow(image1)

    if save_path is None:
        pl.show()
    else:
        pl.savefig(save_path)


def draw_matches(image1, image2, keypoints1, keypoints2, save_path=None, dist=None):
    if torch.is_tensor(image1):
        image1 = image1.cpu().numpy()

    if torch.is_tensor(image2):
        image2 = image2.cpu().numpy()

    if torch.is_tensor(keypoints1):
        keypoints1 = keypoints1.cpu().numpy()
        keypoints2 = keypoints2.cpu().numpy()

    if dist is None:
        inlier_keypoints_left = [cv2.KeyPoint(
            point[0], point[1], 1) for point in keypoints1]
        inlier_keypoints_right = [cv2.KeyPoint(
            point[0], point[1], 1) for point in keypoints2]
        placeholder_matches = [cv2.DMatch(idx, idx, 1)
                               for idx in range(keypoints1.shape[0])]
        image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None,
                                 matchColor=(0, 255, 0))
        #image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None)
    else:
        pass

    width = image3.shape[1]
    height = image3.shape[0]
    pl.figure(figsize=(width, height))
    pl.imshow(image3)
    pl.axis('off')

    if save_path != None:
        fig = pl.gcf()

        # dpi = 300, output = 700*700 pixels
        fig.set_size_inches(7.0/3, 7.0/(width/height)/3)
        pl.gca().xaxis.set_major_locator(pl.NullLocator())
        pl.gca().yaxis.set_major_locator(pl.NullLocator())
        pl.subplots_adjust(top=1, bottom=0, right=1,
                           left=0, hspace=0, wspace=0)
        pl.margins(0, 0)
        fig.savefig(save_path, format='png',
                    transparent=True, dpi=300, pad_inches=0)
        # pl.show()
    else:
        pl.show()

    pl.close()


def draw_matches_v2(img1, cv_kpts1, img2, cv_kpts2, good_matches, mask=None,
                    match_color=(0, 255, 0), pt_color=(255, 0, 0), save_path=None):
    """Draw matches."""
    if type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
        cv_kpts1 = [cv2.KeyPoint(cv_kpts1[i][0], cv_kpts1[i][1], 1)
                    for i in range(cv_kpts1.shape[0])]
        cv_kpts2 = [cv2.KeyPoint(cv_kpts2[i][0], cv_kpts2[i][1], 1)
                    for i in range(cv_kpts2.shape[0])]

    print('Number of matches: %d.' % good_matches.shape[0])
    good_matches = [cv2.DMatch(good_matches[idx, 0], good_matches[idx, 1], 1)
                    for idx in range(good_matches.shape[0])]

    display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches,
                              None,
                              matchColor=match_color,
                              singlePointColor=pt_color,
                              flags=4)  # matchesMask=mask.ravel().tolist(),
    # display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches, None,
    #                             matchColor=(0, 255, 0))

    width = img1.shape[1]*2
    height = img1.shape[0]
    pl.figure(figsize=(width, height))
    pl.imshow(display)
    pl.axis('off')
    if save_path == None:
        pl.show()
    else:
        fig = pl.gcf()
        # dpi = 300, output = 700*700 pixels
        fig.set_size_inches(7.0/3, 7.0/3/(width/height))
        pl.gca().xaxis.set_major_locator(pl.NullLocator())
        pl.gca().yaxis.set_major_locator(pl.NullLocator())
        pl.subplots_adjust(top=1, bottom=0, right=1,
                           left=0, hspace=0, wspace=0)
        pl.margins(0, 0)
        fig.savefig(save_path, format='png',
                    transparent=True, dpi=300, pad_inches=0)
