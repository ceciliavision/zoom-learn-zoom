from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

def align_rigid(image_set, images_gray_set, ref_ind, thre=0.05):
    img_num = len(image_set)
    ref_gray_image = images_gray_set[ref_ind]
    r, c = image_set[0].shape[0:2]

    identity_transform = np.eye(2, 3, dtype=np.float32)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    tform_set_init = [np.eye(2, 3, dtype=np.float32)] * img_num

    tform_set = np.zeros_like(tform_set_init)
    tform_inv_set = np.zeros_like(tform_set_init)
    valid_id = []
    motion_thre = thre * min(r, c)
    for i in range(ref_ind - 1, -1, -1):
        warp_matrix = cv2.estimateRigidTransform(image_uint8(ref_gray_image), 
            image_uint8(images_gray_set[i]), fullAffine=0)
        if warp_matrix is None:
            continue
        tform_set[i] = warp_matrix
        tform_inv_set[i] = cv2.invertAffineTransform(warp_matrix)

        motion_val = abs(warp_matrix - identity_transform).sum()
        if motion_val < motion_thre:
            valid_id.append(i)
        else:
            continue

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    for i in range(ref_ind, img_num, 1):
        warp_matrix = cv2.estimateRigidTransform(image_uint8(ref_gray_image), 
            image_uint8(images_gray_set[i]), fullAffine=0)
        if warp_matrix is None:
            tform_set[i] = identity_transform
            tform_inv_set[i] = identity_transform
            continue
        tform_set[i] = warp_matrix
        tform_inv_set[i] = cv2.invertAffineTransform(warp_matrix)
        motion_val = abs(warp_matrix - identity_transform).sum()
        if motion_val < motion_thre:
            valid_id.append(i)
        else:
            continue
    return tform_set, tform_inv_set, valid_id

def align_ecc(image_set, images_gray_set, ref_ind, thre=0.05):
    img_num = len(image_set)
    ref_gray_image = images_gray_set[ref_ind]
    r, c = image_set[0].shape[0:2]

    warp_mode = cv2.MOTION_AFFINE
    # cv2.MOTION_HOMOGRAPHY # cv2.MOTION_AFFINE # cv2.MOTION_TRANSLATION # cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if  warp_mode == cv2.MOTION_HOMOGRAPHY:
        print("Using homography model for alignment")
        identity_transform = np.eye(3, 3, dtype=np.float32)
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        tform_set_init = [np.eye(3, 3, dtype=np.float32)] * img_num
    else:
        identity_transform = np.eye(2, 3, dtype=np.float32)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        tform_set_init = [np.eye(2, 3, dtype=np.float32)] * img_num

    number_of_iterations = 500
    termination_eps = 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    tform_set = np.zeros_like(tform_set_init)
    tform_inv_set = np.zeros_like(tform_set_init)
    valid_id = []
    motion_thre = thre * min(r, c)
    for i in range(ref_ind - 1, -1, -1):
        _, warp_matrix = cv2.findTransformECC(ref_gray_image, images_gray_set[i], warp_matrix, warp_mode, criteria)
        tform_set[i] = warp_matrix
        tform_inv_set[i] = cv2.invertAffineTransform(warp_matrix)

        motion_val = abs(warp_matrix - identity_transform).sum()
        if motion_val < motion_thre:
            valid_id.append(i)
        else:
            continue

    if  warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    for i in range(ref_ind, img_num, 1):
        _, warp_matrix = cv2.findTransformECC(ref_gray_image, images_gray_set[i], warp_matrix, warp_mode, criteria)
        tform_set[i] = warp_matrix
        tform_inv_set[i] = cv2.invertAffineTransform(warp_matrix)

        motion_val = abs(warp_matrix - identity_transform).sum()
        if motion_val < motion_thre:
            valid_id.append(i)
        else:
            continue
    return tform_set, tform_inv_set, valid_id

def apply_transform(image_set, tform_set, tform_inv_set, t_type, scale=1.):
    tform_set_2 = tform_set
    tform_inv_set_2 = tform_inv_set
    if t_type is None:
        if tform_set[0].shape == 2:
            t_type = "rigid"
        elif tform_set[0].shape == 3:
            t_type = "homography"
        else:
            print("[x] Invalid transforms")
            exit()

    r, c = image_set[0].shape[0:2]
    img_num = len(image_set)
    image_t_set = np.zeros_like(image_set)
    for i in range(img_num):
        image_i = image_set[i]
        tform_i = tform_set[i]
        tform_i_inv = tform_inv_set[i]
        tform_i[0,2] *= scale
        tform_i[1,2] *= scale
        tform_i_inv[0,2] *= scale
        tform_i_inv[1,2] *= scale
        tform_set_2[i] = tform_i
        tform_inv_set_2[i] = tform_i_inv
        if t_type != "homography":
            image_i_transform = cv2.warpAffine(image_i, tform_i, (c, r),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            image_i_transform = cv2.warpPerspective(image_i, tform_i, (c, r),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        image_t_set[i] = image_i_transform

    return image_t_set, tform_set_2, tform_inv_set_2

def sum_aligned_image(image_aligned, image_set):
    sum_img = np.float32(image_set[0]) * 1. / len(image_aligned)
    sum_img_t = np.float32(image_aligned[0]) * 1. / len(image_aligned)
    identity_transform = np.eye(2, 3, dtype=np.float32)
    r, c = image_set[0].shape[0:2]
    for i in range(1, len(image_aligned)):
        sum_img_t += np.float32(image_aligned[i]) * 1. / len(image_aligned)
        image_set_i = cv2.warpAffine(image_set[i], identity_transform, (c, r),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        sum_img += np.float32(image_set_i) * 1. / len(image_aligned)
    return sum_img_t, sum_img
