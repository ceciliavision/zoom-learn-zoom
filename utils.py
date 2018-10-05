from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2, sys, os, rawpy
from PIL import Image
import numpy as np
from timeit import default_timer as timer
import scipy.stats as stats
import tifffile as tiff

######### Local Vars
FOCAL_CODE = 37386
ORIEN_CODE = 274
white_lv = 16383 # 16383 for 14 bits, 4095 for 12 bits
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
def read_input_2x(path_raw, path_process, id_shift=4, is_training=True):
    input_dict = {}
    fileid = int(os.path.basename(path_raw).split('.')[0])
    if is_training:
        if fileid > (7-id_shift):
            return None
    else: # testing on _4
        if fileid != 7-id_shift:# and fileid != 6-id_shift:
            print("Please test with %d or %d"%(7-id_shift, 6-id_shift))
            return None
    path2_raw = path_raw.replace(os.path.basename(path_raw).split('.')[0], "%05d"%(fileid+id_shift))
    path_raw_ref = path_raw.replace(os.path.basename(path_raw).split('.')[0], "%05d"%(1))
    try:
        ratio = 0
        focal1 = readFocal_pil(path_raw)
        focal2 = readFocal_pil(path2_raw)
        focal_ref = readFocal_pil(path_raw_ref)
        if focal2 is None or focal1 is None:
            return None
        ratio = focal1/focal2
        if ratio > 4.5:
            path2_raw = path_raw.replace(os.path.basename(path_raw).split('.')[0], "%05d"%(fileid+id_shift-1))
            try:
                focal2 = readFocal_pil(path2_raw)
            except:
                print('[x] high ratio ; cannot open %s or %s'%(path_raw, path2_raw))
                return None
    except:
        print('[x] Cannot open %s or %s, ratio %s'%(path_raw, path2_raw, ratio))
        return None
    
    tform_txt = os.path.dirname(path_process)+'/tform.txt'
    camera_txt = os.path.dirname(path_process.replace("dslr_10x_both","dslr_10x_both_process"))+'/rawpng/tform_camera.txt'
    wb_txt = os.path.dirname(path_process)+'/wb.txt'
    if not os.path.isfile(tform_txt):
        print('[x] Cannot open %s'%(tform_txt))
        return None
    if not os.path.isfile(wb_txt):
        print('[x] Cannot open %s'%(wb_txt))
        return None
    if not os.path.isfile(camera_txt):
        print('[x] Cannot open %s'%(camera_txt))
        return None
    
    ratio_ref1 = focal_ref/focal1
    ratio_ref2 = focal_ref/focal2
    src_path = path_process.replace(os.path.basename(path_process).split('.')[0].split('_')[0], "%05d"%(fileid+2))
    
    print("Learn a zoom of %s from %s to %s"%(ratio, path2_raw, path_process))
    input_dict['src_path_raw'] = path2_raw
    input_dict['tar_path_raw'] = path2_raw
    input_dict['src_path'] = src_path
    input_dict['tar_path'] = path_process
    input_dict['ratio_ref1'] = ratio_ref1
    input_dict['ratio_ref2'] = ratio_ref2
    input_dict['ratio'] = ratio
    input_dict['src_tform'] = read_tform(tform_txt, key=os.path.basename(path2_raw).split('.')[0])
    input_dict['tar_tform'] = read_tform(tform_txt, key=os.path.basename(path_process).split('.')[0])
    input_dict['src_wb'] = read_wb(wb_txt, key=os.path.basename(path2_raw).split('.')[0]+":")
    input_dict['tar_wb'] = read_wb(wb_txt, key=os.path.basename(path_process).split('.')[0]+":")
    input_dict['camera_tform'] = read_tform(camera_txt, key='00002:')
    return input_dict
    
# 35mm equivalent focal length
def readFocal_pil(image_path):
    if 'ARW' in image_path:
        image_path = image_path.replace('ARW','JPG')
    try:
        img = Image.open(image_path)
    except:
        return None
    exif_data = img._getexif()
    return exif_data[FOCAL_CODE][0]/exif_data[FOCAL_CODE][1]

### CHECK
def readOrien_pil(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    return exif_data[ORIEN_CODE]

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
def read_wb(txtfile, key):
    wb = np.zeros((1,4))
    with open(txtfile) as f:
        for l in f:
            if key in l:
                for i in range(wb.shape[0]):
                    nextline = next(f)
                    try:
                        wb[i,:] = nextline.split()
                    except:
                        print("WB error XXXXXXX")
                        print(txtfile)
    wb = wb.astype(np.float)
    return wb

### CHECK
def get_bayer(path):
    try:
        raw = rawpy.imread(path)
    except:
        return None
    bayer = raw.raw_image_visible.astype(np.float32)
    # bayer_shape = bayer.shape
    # H = bayer_shape[0]
    # W = bayer_shape[1]
    # bayer[0:H:2,0:W:2] *= wb[0,0]
    # bayer[0:H:2,1:W:2] *= wb[0,1]
    # bayer[1:H:2,1:W:2] *= wb[0,3]
    # bayer[1:H:2,0:W:2] *= wb[0,2]
    bayer = (bayer - 512)/ (white_lv - 512) #subtract the black level
    return bayer

def reshape_raw(bayer):
    # print(wb)
    bayer = np.expand_dims(bayer,axis=2) 
    bayer_shape = bayer.shape
    H = bayer_shape[0]
    W = bayer_shape[1]

    reshaped = np.concatenate((bayer[0:H:2,0:W:2,:], 
                       bayer[0:H:2,1:W:2,:],
                       bayer[1:H:2,1:W:2,:],
                       bayer[1:H:2,0:W:2,:]), axis=2)
    return reshaped

def reshape_back_raw(bayer):
    H = bayer.shape[0]
    W = bayer.shape[1]
    newH = int(H*2)
    newW = int(W*2)
    bayer_back = np.zeros((newH, newW))
    bayer_back[0:newH:2,0:newW:2] = bayer[...,0]
    bayer_back[0:newH:2,1:newW:2] = bayer[...,1]
    bayer_back[1:newH:2,1:newW:2] = bayer[...,2]
    bayer_back[1:newH:2,0:newW:2] = bayer[...,3]
    return bayer_back

def write_raw(source_raw, target_raw_path):
    target_raw = rawpy.imread(target_raw_path)
    H, W = source_raw.shape[:2]
    for indi,i in enumerate(range(H)):
        for indj,j in enumerate(range(W)):
            target_raw.raw_image_visible[indi, indj] = source_raw[i, j] * (white_lv - 512) + 512
    rgb = target_raw.postprocess(no_auto_bright=True,
        use_camera_wb=False,
        output_bps=8)
    return rgb

### CHECK
def prepare_input(input_dict, up_ratio=2., mode='train', is_pack=True):
    out_dict = {}
    ratio = input_dict['ratio']
    ratio_offset = up_ratio/ratio ####### HHH #######
    scale_inv_offset = get_scale_matrix(1./ratio_offset)
    scale_offset = get_scale_matrix(ratio_offset)
    scale_ref = get_scale_matrix(input_dict['ratio_ref1'])
    scale_inv_ref = get_scale_matrix(1./input_dict['ratio_ref1'])
    inv_src_tform = cv2.invertAffineTransform(input_dict['src_tform'])
    combined_tform = concat_tform([scale_ref,
        np.append(input_dict['tar_tform'],[[0,0,1]],0),
        np.append(inv_src_tform,[[0,0,1]],0),
        scale_inv_ref,
        scale_inv_offset])
    camera_tform = concat_tform([scale_offset,
        scale_ref,
        np.append(input_dict['camera_tform'], [[0,0,1]],0),
        scale_inv_ref,
        scale_inv_offset])
    inv_tar_tform = cv2.invertAffineTransform(input_dict['tar_tform'])
    combined_tform_src = concat_tform([get_scale_matrix(input_dict['ratio_ref2']),
        np.append(inv_tar_tform,[[0,0,1]],0),
        get_scale_matrix(1./input_dict['ratio_ref2']),
        scale_inv_ref])

    # concat_tform = np.matmul(np.append(input_dict['tar_tform'],[[0,0,1]],0),
    #     np.append(scale_offset,[[0,0,1]],0))
    # concat_tform = np.matmul(np.append(inv_src_tform,[[0,0,1]],0), concat_tform)
    # print("concat_tform",combined_tform)

    input_raw = get_bayer(input_dict['src_path_raw'])
    if input_raw is None:
        with open('./logerror.txt'%(task), 'a') as floss:
            floss.write('error reading raw of: %s\n'%(input_dict['src_path_raw']))
        return None
    input_raw_reshape = reshape_raw(input_raw)
    cropped_raw = crop_fov(input_raw_reshape, 1./input_dict['ratio_ref2'])
    out_dict['input_raw'] = cropped_raw
    try:
        if '.tif' in input_dict['src_path']:
            input_rgb = tiff.imread(os.path.dirname(input_dict['tar_path'])+'/'+
                os.path.basename(input_dict['src_path_raw'].split('.')[0]+'.tif'))
        else:
            input_rgb = Image.open(os.path.dirname(input_dict['tar_path'])+'/'+
                os.path.basename(input_dict['src_path_raw'].split('.')[0]+'.JPG'))
    except Exception as exception:
        print("Failed to open %s"%(os.path.dirname(input_dict['tar_path'])+'/'+
            os.path.basename(input_dict['src_path_raw'].split('.')[0]+'.JPG')))
        return None
    input_rgb = np.array(input_rgb)
    if is_pack:
        input_raw_reshape = reshape_raw(input_raw)
    else:
        input_raw_reshape = input_raw
        
    cropped_input_rgb = crop_fov(input_rgb, 1./input_dict['ratio_ref2'])
    cropped_input_rgb = image_float(cropped_input_rgb)
    out_dict['input_rgb'] = cropped_input_rgb
    out_dict['tform'] = combined_tform[0:2,...]
    out_dict['tform_src'] = combined_tform_src[0:2,...]
    out_dict['camera_tform']  = camera_tform[0:2,...]
    if mode=='inference':
        return out_dict

    tar_raw = get_bayer(input_dict['tar_path_raw'])
    if is_pack:
        tar_raw_reshape = reshape_raw(tar_raw)
    else:
        tar_raw_reshape = tar_raw
    try:
        if '.tif' in input_dict['src_path']:
            tar_rgb = tiff.imread(os.path.dirname(input_dict['tar_path'])+'/'+
                os.path.basename(input_dict['tar_path'].split('.')[0]+'.tif'))
        else:
            tar_rgb = Image.open(os.path.dirname(input_dict['tar_path'])+'/'+
                os.path.basename(input_dict['tar_path'].split('.')[0]+'.JPG'))
    except Exception as exception:
        print("Failed to open %s"%os.path.dirname(input_dict['tar_path'])+'/'+
            os.path.basename(input_dict['tar_path'].split('.')[0]+'.JPG'))
        return None
    tar_rgb = np.array(tar_rgb)
    cropped_rgb = crop_fov(tar_rgb, 1./input_dict['ratio_ref1'])
    tar_raw_reshape = reshape_raw(tar_raw)
    cropped_rgb = image_float(cropped_rgb)
    out_dict['ratio_offset'] = ratio_offset
    out_dict['tar_rgb'] = cropped_rgb
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
def crop_pair(raw, image, croph, cropw, tol=32, raw_tol=4, ratio=2, type='central', fixx=0.5, fixy=0.5):
    is_pad_h = False
    is_pad_w = False
    if type == 'central':
        rand_p = rand_gen.rvs(2)
    elif type == 'uniform':
        rand_p = np.random.rand(2)
    elif type == 'fixed':
        rand_p = [fixx,fixy]
    
    height_raw, width_raw = raw.shape[:2]
    height_rgb, width_rgb = image.shape[:2]
    if croph > height_raw * 2*ratio or cropw > width_raw * 2*ratio:
        print("Image too small to have the specified crop sizes.")
        return None, None
    croph_rgb = croph + tol * 2
    cropw_rgb = cropw + tol * 2
    croph_raw = int(croph/(ratio*2)) + raw_tol*2  # add a small offset to deal with boudary case
    cropw_raw = int(cropw/(ratio*2)) + raw_tol*2  # add a small offset to deal with boudary case
    if croph_rgb > height_rgb:
        sx_rgb = 0
        sx_raw = int(tol/2.)
        is_pad_h = True
        pad_h1_rgb = int((croph_rgb-height_rgb)/2)
        pad_h2_rgb = int(croph_rgb-height_rgb-pad_h1_rgb)
        pad_h1_raw = int(np.ceil(pad_h1_rgb/(2*ratio)))
        pad_h2_raw = int(np.ceil(pad_h2_rgb/(2*ratio)))
    else:
        sx_rgb = int((height_rgb - croph_rgb) * rand_p[0])
        sx_raw = max(0, int((sx_rgb + tol)/(2*ratio)) - raw_tol) # add a small offset to deal with boudary case
    
    if cropw_rgb > width_rgb:
        sy_rgb = 0 
        sy_raw = int(tol/2.)
        is_pad_w = True
        pad_w1_rgb = int((cropw_rgb-width_rgb)/2)
        pad_w2_rgb = int(cropw_rgb-width_rgb-pad_w1_rgb)
        pad_w1_raw = int(np.ceil(pad_w1_rgb/(2*ratio)))
        pad_w2_raw = int(np.ceil(pad_w2_rgb/(2*ratio)))
    else:
        sy_rgb = int((width_rgb - cropw_rgb) * rand_p[1])
        sy_raw = max(0, int((sy_rgb + tol)/(2*ratio)) - raw_tol)
    
    # print("raw cropping params: ", raw.shape, sx_raw, croph_raw, sy_raw, cropw_raw)
    # print("rgb cropping params: ", image.shape, sx_rgb, croph_rgb, sy_rgb, cropw_rgb)
    raw_cropped = raw
    rgb_cropped = image
    if is_pad_h:
        print("Pad h with:", (pad_h1_rgb, pad_h2_rgb),(pad_h1_raw, pad_h2_raw))
        rgb_cropped = np.pad(image, pad_width=((pad_h1_rgb, pad_h2_rgb),(0, 0),(0,0)),
            mode='constant', constant_values=0)
        raw_cropped = np.pad(raw, pad_width=((pad_h1_raw, pad_h2_raw),(0, 0),(0,0)),
            mode='constant', constant_values=0)
    if is_pad_w:
        print("Pad w with:", (pad_w1_rgb, pad_w2_rgb),(pad_w1_raw, pad_w2_raw))
        rgb_cropped = np.pad(image, pad_width=((0, 0),(pad_w1_rgb, pad_w2_rgb),(0,0)),
            mode='constant', constant_values=0)
        raw_cropped = np.pad(raw, pad_width=((0, 0),(pad_w1_raw, pad_w2_raw),(0,0)),
            mode='constant', constant_values=0)
    raw_cropped = raw_cropped[sx_raw:sx_raw+croph_raw, sy_raw:sy_raw+cropw_raw,...]
    rgb_cropped = rgb_cropped[sx_rgb:sx_rgb+croph_rgb, sy_rgb:sy_rgb+cropw_rgb,...]

    return raw_cropped, rgb_cropped

def crop_pair_reverse(raw, image, croph, cropw, tol=32, ratio=2, type='central', fixx=0.5, fixy=0.5):
    is_pad_h = False
    is_pad_w = False
    if type == 'central':
        rand_p = rand_gen.rvs(2)
    elif type == 'uniform':
        rand_p = np.random.rand(2)
    elif type == 'fixed':
        rand_p = [fixx,fixy]
    
    height_raw, width_raw = raw.shape[:2]
    height_rgb, width_rgb = image.shape[:2]
    if croph > height_raw * 2*ratio or cropw > width_raw * 2*ratio:
        print("Image too small to have the specified crop sizes.")
        return None, None
    croph_rgb = croph - tol * 2
    cropw_rgb = cropw - tol * 2
    croph_raw = int(croph/(ratio*2))
    cropw_raw = int(cropw/(ratio*2))
    if croph_raw > height_raw:
        sx_rgb = int(tol)
        sx_raw = 0
        is_pad_h = True
        pad_h1_raw = int((croph_raw-height_raw)/2)
        pad_h2_raw = int(croph_raw-height_raw-pad_h1_raw)
        pad_h1_rgb = int(pad_h1_raw*2*ratio)
        pad_h2_rgb = int(pad_h2_raw*2*ratio)
    else:
        sx_raw = int((height_raw - croph_raw) * rand_p[0])
        sx_rgb = int(sx_raw*2*ratio + tol)
    
    if cropw_raw > width_raw:
        sy_raw = 0 
        sy_rgb = int(tol)
        is_pad_w = True
        pad_w1_raw = int((cropw_raw-width_raw)/2)
        pad_w2_raw = int(cropw_raw-width_raw-pad_w1_raw)
        pad_w1_rgb = int(pad_w1_raw*2*ratio)
        pad_w2_rgb = int(pad_w2_raw*2*ratio)
        
    else:
        sy_raw = int((width_raw - cropw_raw) * rand_p[1])
        sy_rgb = int((sy_raw*2*ratio) + tol)
    
    sy = int((width_raw - cropw) * rand_p[1])
    # print("raw cropping params: ", raw.shape, sx_raw, croph_raw, sy_raw, cropw_raw)
    # print("rgb cropping params: ", image.shape, sx_rgb, croph_rgb, sy_rgb, cropw_rgb)
    raw_cropped = raw
    rgb_cropped = image
    if is_pad_h:
        print("Pad h with:", (pad_h1_rgb, pad_h2_rgb),(pad_h1_raw, pad_h2_raw))
        rgb_cropped = np.pad(image, pad_width=((pad_h1_rgb, pad_h2_rgb),(0, 0),(0,0)),
            mode='constant', constant_values=0)
        raw_cropped = np.pad(raw, pad_width=((pad_h1_raw, pad_h2_raw),(0, 0),(0,0)),
            mode='constant', constant_values=0)
    if is_pad_w:
        print("Pad w with:", (pad_w1_rgb, pad_w2_rgb),(pad_w1_raw, pad_w2_raw))
        rgb_cropped = np.pad(image, pad_width=((0, 0),(pad_w1_rgb, pad_w2_rgb),(0,0)),
            mode='constant', constant_values=0)
        raw_cropped = np.pad(raw, pad_width=((0, 0),(pad_w1_raw, pad_w2_raw),(0,0)),
            mode='constant', constant_values=0)
    raw_cropped = raw_cropped[sx_raw:sx_raw+croph_raw, sy_raw:sy_raw+cropw_raw,...]
    rgb_cropped = rgb_cropped[sx_rgb:sx_rgb+croph_rgb, sy_rgb:sy_rgb+cropw_rgb,...]

    return raw_cropped, rgb_cropped

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
def post_process_rgb(target_rgb, out_size, tform):
    target_rgb_warp = cv2.warpAffine(target_rgb, tform, (out_size[0], out_size[1]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    transformed_corner = get_transformed_corner(tform, out_size[0], out_size[1])
    target_rgb_process = target_rgb_warp[transformed_corner['minw']:transformed_corner['maxw'],
        transformed_corner['minh']:transformed_corner['maxh'],:]
    return target_rgb_process, transformed_corner

### CHECK
# image_set: a list of images
def bgr_gray(image_set, color='rgb'):
    img_num = len(image_set)
    image_gray_set = []
    for i in range (img_num):
        if color == 'rgb':
            image_gray_i = cv2.cvtColor(image_set[i], cv2.COLOR_RGB2GRAY)
        elif color == 'bgr':
            image_gray_i = cv2.cvtColor(image_set[i], cv2.COLOR_BGR2GRAY)
        image_gray_set.append(image_gray_i)
    return image_gray_set

### CHECK
def image_float(image):
    if image.max() < 2:
        return image.astype(np.float32)
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

def clipped(image):
    if image.max() <= 10:
        return np.minimum(np.maximum(image,0.0),1.0)
    else:
        return np.minimum(np.maximum(image,0.0),255.0)

def postprocess_output(image_set, is_align=False, is_color=False):
    image_set_processed = image_set
    if is_align:
        print("Running alignment")
        image_ref_processed = image_set[0]
        height, width = image_ref_processed.shape[0:2]
        corner = np.array([[0,0,width,width],[0,height,0,height],[1,1,1,1]])

        image_set_gray = bgr_gray(image_set,'rgb')
        t, t_inv, valid_id = align_ecc(image_set, image_set_gray, 0, thre=0.2)
        images_set_t, t, t_inv = apply_transform(image_set, t, t_inv, 'ECC', scale=1)
        
        for i in range(2):
            corner_out = np.matmul(np.vstack([np.array(t_inv[i]),[0,0,1]]),corner)
            # print(i, corner_out)
            corner_out[0,:] = np.divide(corner_out[0,:],corner_out[2,:])
            corner_out[1,:] = np.divide(corner_out[1,:],corner_out[2,:])
            corner_out = corner_out[..., np.newaxis]
            if i == 0:
                corner_t = corner_out
            else:
                corner_t = np.append(corner_t,corner_out,2)
        min_w = np.max(corner_t[0,[0,1],:])
        min_w = int(np.max(np.ceil(min_w),0))
        min_h = np.max(corner_t[1,[0,2],:])
        min_h = int(np.max(np.ceil(min_h),0))
        max_w = np.min(corner_t[0,[2,3],:])
        max_w = int(np.floor(max_w))
        max_h = np.min(corner_t[1,[1,3],:])
        max_h = int(np.floor(max_h))

        image_set_processed = []
        for i in range(len(image_set)):
            image_set_processed.append(images_set_t[i][min_h:max_h,min_w:max_w,:])
    if is_color:
        print("Color matching")
        image_ref_processed = image_set_processed[0]
        image_set_tmp = []
        image_set_tmp.append(image_ref_processed)
        for i in range(1,len(image_set)):
            image_out_processed = image_set_processed[i]
            image_tmp = np.zeros_like(image_out_processed)
            for c in range(image_ref_processed.shape[2]):
                image_tmp[:, :, c] = hist_match(image_out_processed[:, :, c], image_ref_processed[:, :, c])
            image_set_tmp.append(image_tmp)
        image_set_processed = image_set_tmp
    return image_set_processed

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.

    Code adapted from
    http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

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

def apply_transform_single(image, tform, c, r):
    return cv2.warpAffine(image, tform, (c, r),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

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
def unaligned_loss(prediction, target, tar_w, tar_h, tol, stride=1):
    min_error = 1000000
    shifted_loss = np.zeros((int(tol*2/stride), int(tol*2/stride)))
    for idi,i in enumerate(range(0,(tol*2),stride)):
        for idj,j in enumerate(range(0,(tol*2),stride)):
            canvas = np.zeros((tar_w, tar_h, prediction.shape[-1]))
            canvas[i:i+prediction.shape[0],j:j+prediction.shape[1],:] = prediction
            shifted_loss[idi,idj] = (abs((target-canvas)[i:i+prediction.shape[0],j:j+prediction.shape[1],:])).mean()
            print(i,j,shifted_loss[idi,idj])
            if shifted_loss[idi,idj] < min_error:
                image_min = canvas
                min_error = shifted_loss[idi,idj]
    loss = shifted_loss.min()
    mini, minj = np.unravel_index(shifted_loss.argmin(), shifted_loss.shape)
    return image_min, loss, mini*stride, minj*stride

### CHECK
def apply_gamma(image, gamma=2.22,is_apply=True):
    if not is_apply:
        image[image < 0] = 0.
        return image
    if image.max() > 5:
        image = image_float(image)
    if image.min() < 0:
        print("Negative values in images, zero out")
        image[image < 0] = 0.
    image_copy = image
    # image_copy[image < 0.0031308] *= 4.5
    image_copy = image_copy ** (1./gamma)
    return image_copy

