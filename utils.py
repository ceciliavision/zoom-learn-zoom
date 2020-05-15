from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2, sys, os, rawpy
from PIL import Image
import numpy as np
import scipy.stats as stats

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
# generate random numbers for random crop
rand_gen = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

######### Util functions
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_raw_file(filename):
    return any(filename.endswith(extension) for extension in RAW_EXTENSIONS)

def read_wb_lv(device):
    if device == "sony":
        white_lv = 16383
        black_lv = 512
    elif device == "iphone":
        white_lv = 4367
        black_lv = 528
    else:
        print("Unknow device, please change or add your own.")
        exit()
    return white_lv, black_lv

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
    corner = np.eye(4, dtype=np.float32)
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
            if 'corner' in l:
                nextline = next(f)
                corner = nextline.split()
    return tform, corner

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

### CHECK
def compute_wb(raw_path):
    print("Computing WB for %s"%(raw_path))
    bayer = rawpy.imread(raw_path)
    rgb_nowb = bayer.postprocess(gamma=(1, 1),
        no_auto_bright=True,
        use_camera_wb=False,
        output_bps=16)

    rgb_wb = bayer.postprocess(gamma=(1, 1),
        no_auto_bright=True,
        use_camera_wb=True,
        output_bps=16)

    scale=[np.mean(rgb_wb[...,0])/np.mean(rgb_nowb[...,0]), 
        np.mean(rgb_wb[...,1])/np.mean(rgb_nowb[...,1]),
        np.mean(rgb_wb[...,1])/np.mean(rgb_nowb[...,1]),
        np.mean(rgb_wb[...,2])/np.mean(rgb_nowb[...,2])]
    wb = np.zeros((1,4))
    wb[0,0] = scale[0]
    wb[0,1] = scale[1]
    wb[0,2] = scale[2]
    wb[0,3] = scale[3]
    return wb

def make_mosaic(im, mosaic_type='bayer'):
    H, W=im.shape[:2]
    mosaic=np.zeros((H, W))
    mosaic[0:H:2, 0:W:2] = im[0:H:2, 0:W:2, 0]
    mosaic[0:H:2, 1:W:2] = im[0:H:2, 1:W:2, 1]
    mosaic[1:H:2, 0:W:2] = im[1:H:2, 0:W:2, 1]
    mosaic[1:H:2, 1:W:2] = im[1:H:2, 1:W:2, 2]
    return mosaic

def add_noise(im):
    sz = im.shape
    noise_level = np.random.rand(1)
    noise_level *= 0.0784
    noise_level += 0.0000
    noise = noise_level*np.random.randn(sz[0], sz[1])
    im += noise
    return im, noise_level

### CHECK
def get_bayer(path, black_lv, white_lv):
    try:
        raw = rawpy.imread(path)
    except:
        return None
    bayer = raw.raw_image_visible.astype(np.float32)
    bayer = (bayer - black_lv)/ (white_lv - black_lv) #subtract the black level
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

def write_raw(source_raw, target_raw_path, device='sony'):
    white_lv, black_lv = read_wb_lv(device)
    target_raw = rawpy.imread(target_raw_path)
    H, W = source_raw.shape[:2]
    for indi,i in enumerate(range(H)):
        for indj,j in enumerate(range(W)):
            target_raw.raw_image_visible[indi, indj] = source_raw[i, j] * (white_lv - black_lv) + black_lv
    rgb = target_raw.postprocess(no_auto_bright=True,
        use_camera_wb=False,
        output_bps=8)
    return rgb

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
def crop_fov_free(image, ratio, crop_fracx=1./2, crop_fracy=1./2):
    width, height = image.shape[:2]
    new_width = width * ratio
    new_height = height * ratio
    left = np.ceil((width - new_width) * crop_fracx)
    top = np.ceil((height - new_height) * crop_fracy)
    # right = np.floor((width + new_width) * crop_frac)
    # bottom = np.floor((height + new_height) * crop_frac)
    # print("Cropping boundary: ", top, bottom, left, right)
    cropped = image[int(left):int(left+new_width), int(top):int(top+new_height), ...]
    return cropped

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

### CHECK
# use PIL image resize
def resize_pil(image, ratio):
    image = Image.fromarray(image)
    image = image.resize((int(image.width*ratio),
                             int(image.height*ratio)),
                             Image.ANTIALIAS)
    return np.array(image)

def clipped(image):
    if image.max() <= 10:
        return np.minimum(np.maximum(image,0.0),1.0)
    else:
        return np.minimum(np.maximum(image,0.0),255.0)

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
    image_copy = image_copy ** (1./gamma)
    return image_copy

# Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
def rgb2yuv(rgb):
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp

def get_scale_matrix(ratio):
    scale = np.eye(3, 3, dtype=np.float32)
    scale[0,0] = ratio
    scale[1,1] = ratio
    return scale

def concat_tform(tform_list):
    tform_c = tform_list[0]
    for tform in tform_list[1:]:
        tform_c = np.matmul(tform, tform_c)
    return tform_c

def warp_image(target_rgb, out_size, tform):
    target_rgb_warp = cv2.warpAffine(target_rgb, tform, (out_size[0], out_size[1]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    transformed_corner = get_transformed_corner(tform, out_size[0], out_size[1])
    target_rgb_process = target_rgb_warp[transformed_corner['minw']:transformed_corner['maxw'],
        transformed_corner['minh']:transformed_corner['maxh'],:]
    return target_rgb_process, transformed_corner

# utils to prepare aligned rgb-raw paires
def crop_pair(
    raw, image,
    croph, cropw,
    tol=32, raw_tol=4, ratio=2,
    type='central',
    fixx=0.5, fixy=0.5):

    is_pad_h = False
    is_pad_w = False
    if type == 'central':
        rand_p = rand_gen.rvs(2)
    elif type == 'random':
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

def concat_tform(tform_list):
    tform_c = tform_list[0]
    for tform in tform_list[1:]:
        tform_c = np.matmul(tform, tform_c)
    return tform_c

def post_process_rgb(target_rgb, out_size, tform):
    target_rgb_warp = cv2.warpAffine(target_rgb, tform, (out_size[1], out_size[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    transformed_corner = get_transformed_corner(tform, out_size[0], out_size[1])
    target_rgb_process = target_rgb_warp[transformed_corner['minh']:transformed_corner['maxh'],
        transformed_corner['minw']:transformed_corner['maxw'],:]
    return target_rgb_process, transformed_corner

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
