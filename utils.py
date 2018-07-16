from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2, sys, os, rawpy
from PIL import Image
import numpy as np
from timeit import default_timer as timer
import scipy.stats as stats
import tifffile

######### Local Vars
FOCAL_CODE = 37386
ORIEN_CODE = 274
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', 'tiff',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
RAW_EXTENSIONS = [
    '.ARW', '.arw', '.CR2', 'cr2',
]
lower, upper = 0., 1.
mu, sigma = 0.5, 0.2
rand_gen = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

crop_sz = 512

######### Util functions
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_raw_file(filename):
    return any(filename.endswith(extension) for extension in RAW_EXTENSIONS)

def read_paths(path, type='RAW'):
    paths=[]
    for dirname in path:
        for root, _, fnames in sorted(os.walk(dirname)):
            for fname in sorted(fnames):
                if type == 'RAW':
                    if is_raw_file(fname):
                        paths.append(os.path.join(root, fname))
                else:
                    if is_image_file(fname):
                        paths.append(os.path.join(root, fname))
    return paths

def read_input_2x(path):
    input_dict = {}
    fileid = int(os.path.basename(path).split('.')[0])
    path2 = path.replace(os.path.basename(path).split('.')[0], "%05d"%(fileid+2))
    path_ref = path.replace(os.path.basename(path).split('.')[0], "%05d"%(1))
    try:
        focal1 = readFocal_pil(path)
        focal2 = readFocal_pil(path2)
        focal_ref = readFocal_pil(path_ref)
    except:
        print('[x] Cannot open %s or %s'%(path, path2))
        return None
    
    ratio = focal1/focal2
    ratio_ref1 = focal_ref/focal1
    ratio_ref2 = focal_ref/focal2
    src_path = path2
    tar_path = path
    
    print("Learn a zoom of %s from %s to %s"%(ratio, src_path, tar_path))
    input_dict['src_path'] = src_path
    input_dict['tar_path'] = tar_path
    input_dict['ratio_ref1'] = ratio_ref1
    input_dict['ratio_ref2'] = ratio_ref2
    input_dict['ratio'] = ratio
    input_dict['src_tform'] = get_tform(os.path.dirname(src_path)+'/tform.txt',
        key=os.path.basename(src_path).split('.')[0])
    input_dict['tar_tform'] = get_tform(os.path.dirname(tar_path)+'/tform.txt',
        key=os.path.basename(tar_path).split('.')[0])
    return input_dict
    
def get_bayer(path):
    raw = rawpy.imread(path)
    bayer = raw.raw_image_visible.astype(np.float32)
    bayer = np.maximum(bayer - 512,0)/ (16383 - 512) #subtract the black level
    return bayer

def reshape_raw(bayer):
    bayer = np.expand_dims(bayer,axis=2) 
    bayer_shape = bayer.shape
    H = bayer_shape[0]
    W = bayer_shape[1]

    reshaped = np.concatenate((bayer[0:H:2,0:W:2,:], 
                       bayer[0:H:2,1:W:2,:],
                       bayer[1:H:2,1:W:2,:],
                       bayer[1:H:2,0:W:2,:]), axis=2)
    return reshaped

def prepare_input(input_dict, tw=512, th=512, pw=512, ph=512, pre_crop=False):
    out_dict = {}
    ratio = input_dict['ratio']
    ratio_offset = 2./ratio
    scale_inv_offset = get_scale(1./ratio_offset)
    scale_ref = get_scale(input_dict['ratio_ref1'])
    scale_inv_ref = get_scale(1./input_dict['ratio_ref1'])
    inv_src_tform = cv2.invertAffineTransform(input_dict['src_tform'])
    combined_tform = concat_tform([scale_ref,
        np.append(inv_src_tform,[[0,0,1]],0),
        np.append(input_dict['tar_tform'],[[0,0,1]],0),
        scale_inv_ref,
        scale_inv_offset])

    # concat_tform = np.matmul(np.append(input_dict['tar_tform'],[[0,0,1]],0),
    #     np.append(scale_offset,[[0,0,1]],0))
    # concat_tform = np.matmul(np.append(inv_src_tform,[[0,0,1]],0), concat_tform)
    print("concat_tform",combined_tform)

    input_raw = get_bayer(input_dict['src_path'])
    tar_raw = get_bayer(input_dict['tar_path'])
    tar_rgb = tifffile.imread(os.path.dirname(input_dict['tar_path'])+'/rawjpg/'+os.path.basename(input_dict['tar_path'].split('.')[0]+'.tiff'))
    input_raw_reshape = reshape_raw(input_raw)
    cropped_raw = crop_fov(input_raw_reshape, 1./input_dict['ratio_ref2'])
    cropped_rgb = crop_fov(tar_rgb, 1./input_dict['ratio_ref1'])
    tar_raw_reshape = reshape_raw(tar_raw)

    if pre_crop:
        input_raw, cropped_rgb = crop_raw_image(input_raw, cropped_rgb, 512, 512, type='central')
        if input_raw is None or cropped_rgb is None:
            return None, None
    cropped_rgb = np.array(cropped_rgb,dtype=np.float32) / (255. * 255.)
    cropped_rgb = np.expand_dims(cropped_rgb, axis=0)
    input_raw = np.expand_dims(cropped_raw, axis=0)
    out_dict['ratio_offset'] = ratio_offset
    out_dict['input_raw'] = input_raw
    out_dict['tar_rgb'] = cropped_rgb
    out_dict['tform'] = combined_tform[0:2,...]
    return out_dict

def get_scale(ratio):
    scale = np.eye(3, 3, dtype=np.float32)
    scale[0,0] = ratio
    scale[1,1] = ratio
    return scale

def concat_tform(tform_list):
    tform_c = tform_list[0]
    for tform in tform_list[1:]:
        tform_c = np.matmul(tform, tform_c)
    return tform_c

# 35mm equivalent focal length
def readFocal_pil(image_path):
    if 'ARW' in image_path:
        image_path = image_path.replace('ARW','JPG')
    img = Image.open(image_path)
    exif_data = img._getexif()
    return exif_data[FOCAL_CODE][0]/exif_data[FOCAL_CODE][1]

def readOrien_pil(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    return exif_data[ORIEN_CODE]

# PIL image format
def crop_raw_image(raw, image, croph, cropw, type='central'):
    height, width = raw.shape[:2]
    if croph > height or cropw > width:
        print("Image too small to have the specified crop sizes.")
        return None
    if type == 'central':
        rand_p = rand_gen.rvs(2)
        sx = int((height - croph) * rand_p[0])
        sy = int((width - cropw) * rand_p[1])
        area = (sy*2, sx*2, sy*2+cropw*2, sx*2+croph*2)
    return raw[sx:sx+croph, sy:sy+cropw,...], image.crop(area)

def crop_fov(image, ratio):
    width, height = image.shape[:2]
    new_width = width * ratio
    new_height = height * ratio
    left = np.ceil((width - new_width)/2.)
    top = np.ceil((height - new_height)/2.)
    right = np.floor((width + new_width)/2.)
    bottom = np.floor((height + new_height)/2.)
    # print("Cropping boundary: ", top, bottom, left, right)
    cropped = image[int(left):int(right), int(top):int(bottom), ...]
    return cropped

# image_set: a list of images
def bgr_gray(image_set):
    img_num = len(image_set)
    image_gray_set = []
    for i in range (img_num):
        image_gray_i = cv2.cvtColor(image_set[i], cv2.COLOR_BGR2GRAY)
        image_gray_set.append(image_gray_i)
    return image_gray_set

def image_float(image):
    image = image.astype(np.float32) / 255
    return image

def image_uint8(image):
    if image.max() > 10:
        return image
    image = (image * 255).astype(np.uint8)
    return image

def get_tform(txtfile, key, model='ECC'):
    if model in ['ECC', 'RIGID']:
        tform = np.eye(2, 3, dtype=np.float32)
    else:
        tform = np.eye(3, 3, dtype=np.float32)
    with open(txtfile) as f:
        for l in f:
            if key in l:
                for i in range(tform.shape[0]):
                    nextline = next(f)
                    tform[i,:] = nextline.split()
    return tform

def get_feature(image, feature_type):
    # Initiate detector
    if feature_type is "ORB":
        feat = cv2.ORB_create()
    elif feature_type is "SURF":
        feat = cv2.SURF(400)
    else:
        feat = cv2.SIFT()

    # find the keypoints
    key_pts = feat.detect(image, None)
    # compute the descriptors with ORB
    key_pts, key_descriptors = feat.compute(image, key_pts)  # key_pts correspondes to struct in MATLAB
    return key_descriptors, key_pts

# check
def get_transform(ref_descriptors, ref_pts, image_set, t_type="rigid", feature_type='ORB'):
    img_num = len(image_set)
    tform_set = []
    tform_inv_set = []
    for i in range (img_num):
        base_image = image_set[i]
        query_pts, query_descriptors = get_feature(base_image, "ORB")
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_features = bf_matcher.match(query_descriptors, ref_descriptors) # query to train
        avg_distance = 1e-6 + sum(m.distance for m in matches_features)/len(matches_features)
        matches_good = []
        for m in matches_features:
            if m.distance < avg_distance * 1. / 3:
                matches_good.append(m)

        num_matches_good = len(matches_good)
        print("Extracted %d good matches" % num_matches_good)
        query_match_pts = np.array(list(query_pts[m.queryIdx].pt for m in matches_good))
        ref_match_pts = np.array(list(ref_pts[m.trainIdx].pt for m in matches_good))
        ratio = max(ref_match_pts.max(), query_match_pts.max())//255 + 1
        query_match_pts_scale = (query_match_pts/ratio).astype(np.uint8).reshape(2,num_matches_good)
        ref_match_pts_scale = (ref_match_pts/ratio).astype(np.uint8).reshape(2,num_matches_good)

        if t_type == "rigid":
            tform = cv2.estimateRigidTransform(query_match_pts_scale, ref_match_pts_scale, True) # query to train
        elif t_type == "homography":
            tform, status = cv2.findHomography(query_match_pts, ref_match_pts)
        else:
            tform = cv2.estimateRigidTransform(query_match_pts_scale, ref_match_pts_scale, False)

        # print(tform)
        tform_set.append(tform)
        tform_inv_set.append(cv2.invertAffineTransform(tform))
    return tform_set, tform_inv_set

# check
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

# images don't need to have same size
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
        # print("warp_matrix: ", warp_matrix)
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
    # select the image as reference
    # ref_image = image_set[ref_ind]
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
