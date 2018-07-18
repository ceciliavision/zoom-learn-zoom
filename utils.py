from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2, sys, os, rawpy
from PIL import Image
import numpy as np
from timeit import default_timer as timer
import scipy.stats as stats
# import tifffile

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

# read in input dict of image pairs with 2X zoom
def read_input_2x(path_raw, path_process):
    input_dict = {}
    fileid = int(os.path.basename(path_raw).split('.')[0])
    if fileid >= 5:
        return None
    path2_raw = path_raw.replace(os.path.basename(path_raw).split('.')[0], "%05d"%(fileid+2))
    path_raw_ref = path_raw.replace(os.path.basename(path_raw).split('.')[0], "%05d"%(1))
    try:
        focal1 = readFocal_pil(path_raw)
        focal2 = readFocal_pil(path2_raw)
        focal_ref = readFocal_pil(path_raw_ref)
        ratio = focal1/focal2
        if ratio > 2.5:
            path2_raw = path_raw.replace(os.path.basename(path_raw).split('.')[0], "%05d"%(fileid+2))
            try:
                focal2 = readFocal_pil(path2_raw)
            except:
                print('[x] Cannot open %s or %s'%(path_raw, path2_raw))
                return None
    except:
        print('[x] Cannot open %s or %s'%(path_raw, path2_raw))
        return None
    
    if not os.path.isfile(os.path.dirname(path_process)+'/tform.txt'):
        print('[x] Cannot open %s'%(os.path.dirname(path_process)+'/tform.txt'))
        return None
    
    ratio_ref1 = focal_ref/focal1
    ratio_ref2 = focal_ref/focal2
    src_path = path2_raw
    tar_path = path_process
    
    print("Learn a zoom of %s from %s to %s"%(ratio, src_path, tar_path))
    input_dict['src_path'] = src_path
    input_dict['tar_path_raw'] = path2_raw
    input_dict['tar_path'] = tar_path
    input_dict['ratio_ref1'] = ratio_ref1
    input_dict['ratio_ref2'] = ratio_ref2
    input_dict['ratio'] = ratio
    input_dict['src_tform'] = read_tform(os.path.dirname(tar_path)+'/tform.txt',
        key=os.path.basename(src_path).split('.')[0])
    input_dict['tar_tform'] = read_tform(os.path.dirname(tar_path)+'/tform.txt',
        key=os.path.basename(tar_path).split('.')[0])
    return input_dict
    
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

### CHECK
def prepare_input(input_dict, pw=512, ph=512, tol=32, pre_crop=False):
    out_dict = {}
    ratio = input_dict['ratio']
    ratio_offset = 2./ratio ####### HHH #######
    scale_inv_offset = get_scale_matrix(1./ratio_offset)
    scale_ref = get_scale_matrix(input_dict['ratio_ref1'])
    scale_inv_ref = get_scale_matrix(1./input_dict['ratio_ref1'])
    inv_src_tform = cv2.invertAffineTransform(input_dict['src_tform'])
    combined_tform = concat_tform([scale_ref,
        np.append(inv_src_tform,[[0,0,1]],0),
        np.append(input_dict['tar_tform'],[[0,0,1]],0),
        scale_inv_ref,
        scale_inv_offset])

    # concat_tform = np.matmul(np.append(input_dict['tar_tform'],[[0,0,1]],0),
    #     np.append(scale_offset,[[0,0,1]],0))
    # concat_tform = np.matmul(np.append(inv_src_tform,[[0,0,1]],0), concat_tform)
    # print("concat_tform",combined_tform)

    input_raw = get_bayer(input_dict['src_path'])
    tar_raw = get_bayer(input_dict['tar_path_raw'])
    tar_rgb = Image.open(os.path.dirname(input_dict['tar_path'])+'/'+os.path.basename(input_dict['tar_path'].split('.')[0]+'.png'))
    tar_rgb = np.array(tar_rgb)
    input_raw_reshape = reshape_raw(input_raw)
    cropped_raw = crop_fov(input_raw_reshape, 1./input_dict['ratio_ref2'])
    cropped_rgb = crop_fov(tar_rgb, 1./input_dict['ratio_ref1'])
    tar_raw_reshape = reshape_raw(tar_raw)

    cropped_rgb = image_float(cropped_rgb)
    cropped_rgb = np.expand_dims(cropped_rgb, axis=0)
    cropped_raw = np.expand_dims(cropped_raw, axis=0)
    out_dict['ratio_offset'] = ratio_offset
    out_dict['input_raw'] = cropped_raw
    out_dict['tar_rgb'] = cropped_rgb
    out_dict['tform'] = combined_tform[0:2,...]
    return out_dict

### CHECK
def get_scale_matrix(ratio):
    scale = np.eye(3, 3, dtype=np.float32)
    scale[0,0] = ratio
    scale[1,1] = ratio
    return scale

### CHECK
def concat_tform(tform_list):
    tform_c = tform_list[0]
    for tform in tform_list[1:]:
        tform_c = np.matmul(tform, tform_c)
    return tform_c

# PIL image format
def crop_pair(raw, image, croph, cropw, tol=32, ratio=2, type='central'):
    if type == 'central':
        rand_p = rand_gen.rvs(2)
    elif type == 'uniform':
        rand_p = np.random.rand(2)
    
    height_raw, width_raw = raw.shape[:2]
    height_rgb, width_rgb = image.shape[:2]
    if croph > height_raw * 2*ratio or cropw > width_raw * 2*ratio:
        print("Image too small to have the specified crop sizes.")
        return None, None
    croph_rgb = croph + tol * 2
    cropw_rgb = cropw + tol * 2
    croph_raw = int(croph/(ratio*2))
    cropw_raw = int(cropw/(ratio*2))
    if croph_rgb > height_rgb:
        sx_rgb = 0
        sx_raw = int(tol/2.)
    else:
        sx_rgb = int((height_rgb - croph_rgb) * rand_p[0])
        sx_raw = int((sx_rgb + tol)/(2*ratio))
    
    if cropw_rgb > width_rgb:
        sy_rgb = 0 
        sy_raw = int(tol/2.)
    else:
        sy_rgb = int((width_rgb - cropw_rgb) * rand_p[1])
        sy_raw = int((sy_rgb + tol)/(2*ratio))
    
    sy = int((width_raw - cropw) * rand_p[1])
    # print("raw cropping params: ", raw.shape, sx_raw, croph_raw, sy_raw, cropw_raw)
    # print("rgb cropping params: ", image.shape, sx_rgb, croph_rgb, sy_rgb, cropw_rgb)
    raw_cropped = raw[sx_raw:sx_raw+croph_raw, sy_raw:sy_raw+cropw_raw,...]
    image_cropped = image[sx_rgb:sx_rgb+croph_rgb, sy_rgb:sy_rgb+cropw_rgb,...]
    return raw_cropped, image_cropped

### CHECK
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

### CHECK
# image_set: a list of images
def bgr_gray(image_set):
    img_num = len(image_set)
    image_gray_set = []
    for i in range (img_num):
        image_gray_i = cv2.cvtColor(image_set[i], cv2.COLOR_BGR2GRAY)
        image_gray_set.append(image_gray_i)
    return image_gray_set

### CHECK
def image_float(image):
    if image.dtype is np.dtype(np.uint16):
        image = image.astype(np.float32) / (255*255)
    elif image.dtype is np.dtype(np.uint8):
        image = image.astype(np.float32) / 255
    return image

### CHECK
def image_uint8(image):
    if image.max() > 10:
        return image
    image = (image * 255).astype(np.uint8)
    return image

### CHECK
def read_tform(txtfile, key, model='ECC'):
    if model in ['ECC', 'RIGID']:
        tform = np.eye(2, 3, dtype=np.float32)
    else:
        tform = np.eye(3, 3, dtype=np.float32)
    with open(txtfile) as f:
        for l in f:
            if "00001-"+key in l:
                for i in range(tform.shape[0]):
                    nextline = next(f)
                    tform[i,:] = nextline.split()
    return tform

### CHECK
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

### CHECK 
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

### CHECK
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

### CHECK
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

### CHECK
def get_transformed_corner(tform, h, w):
    corner = np.array([[0,0,w,w],[0,h,0,h],[1,1,1,1]])
    inv_tform = cv2.invertAffineTransform(tform)
    corner_t = np.matmul(np.vstack([np.array(inv_tform),[0,0,1]]),corner)
    min_w = np.max(corner_t[0,[0,1]])
    min_w = int(np.max(np.ceil(min_w),0))
    min_h = np.max(corner_t[1,[0,2]])
    min_h = int(np.max(np.ceil(min_h),0))
    max_w = np.min(corner_t[0,[2,3]])
    max_w = int(np.floor(max_w))
    max_h = np.min(corner_t[1,[1,3]])
    max_h = int(np.floor(max_h))
    tformed = {}
    tformed['minw'] = max(0,min_w)
    tformed['maxw'] = min(w,max_w)
    tformed['minh'] = max(0,min_h)
    tformed['maxh'] = min(h,max_h)
    return tformed

### CHECK
def post_process_rgb(target_rgb, out_size, tform):
    target_rgb_warp = cv2.warpAffine(target_rgb, tform, (out_size[0], out_size[1]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    transformed_corner = get_transformed_corner(tform, out_size[0], out_size[1])
    target_rgb_process = target_rgb_warp[transformed_corner['minw']:transformed_corner['maxw'],
        transformed_corner['minh']:transformed_corner['maxh'],:]
    return target_rgb_process, transformed_corner