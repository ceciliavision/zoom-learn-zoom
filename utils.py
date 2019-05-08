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
        if os.path.isdir(dirname):
            for root, _, fnames in sorted(os.walk(dirname)):
                for fname in sorted(fnames):
                    if type == 'RAW':
                        if is_raw_file(fname):
                            paths.append(os.path.join(root, fname))
                    else:
                        if is_image_file(fname):
                            paths.append(os.path.join(root, fname))
        else:
            if type == 'RAW':
                if is_raw_file(dirname):
                    paths.append(os.path.join(path, dirname))
            else:
                if is_image_file(fname):
                    paths.append(os.path.join(path, dirname))
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

