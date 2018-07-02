import cv2,os,rawpy,struct
import rawpy.enhance
import numpy as np
from imageio import imread, imwrite, imsave
import scipy.io
from skimage import io

path = '/home/xuanerzh/Downloads/burst/raw/'
allfiles=os.listdir(path)
imlist=[filename for filename in allfiles if filename[-4:] in [".ARW"]]
N=len(imlist)

save_path = path + "processed/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

arr=0

output = rawpy.imread(path+imlist[0])
for im in imlist:
    file = path+im
    print("processing file: ",file)
    raw = rawpy.imread(file)
    bad_pixels = rawpy.enhance.find_bad_pixels([file])
    rawpy.enhance.repair_bad_pixels(raw, bad_pixels, method='median')
    imarr = raw.raw_image_visible.astype(np.float32)
    arr = arr+imarr/N

w,h = imarr.shape
for i in range(w):
    for j in range(h):
        output.raw_image_visible[i, j] = arr[i, j]
output = output.postprocess(use_camera_wb=True)

# print("daylight white balance: ", raw.daylight_whitebalance)
# print("camera white balance: ", raw.camera_whitebalance)
# print("black level: ",raw.black_level_per_channel)

print(output.min(), output.max(), output.shape)
imsave(save_path+"raw_average.png",np.uint8(output))
