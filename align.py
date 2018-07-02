from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2,os,argparse
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="/home/xuanerzh/Downloads/compare/",
                    required=True, help="root folder that contains the images")
parser.add_argument("--ext", default='.JPG', help="file name txt file")
parser.add_argument("--filetxt", default=None, help="file name txt file")
ARGS = parser.parse_args()
print(ARGS)

align = True
folder = ARGS.folder
MOTION_MODEL = "ECC"
allfiles=os.listdir(folder)
imlist=[filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG",".png",".PNG"]]

out_f = os.path.join(folder, 'aligned')
out_sum = os.path.join(folder, 'compare')
if not os.path.exists(out_f):
    os.mkdir(out_f)
if not os.path.exists(out_sum):
    os.mkdir(out_sum)

images = []

with open(ARGS.filetxt) as f:
    for l in f:
        line = l.splitlines()[0]
        img_rgb = cv2.imread(ARGS.folder + line)
        print(ARGS.folder + line)
        img_rgb = utils.ImageToFloat(img_rgb)  # normalize to [0, 1]
        images.append(img_rgb)

num_img = len(images)

# operate on downsampled images
ref_ind = 0

#################### ALIGN ####################
height, width = img_rgb.shape[0:2]
corner = np.array([[0,0,width,width],[0,height,0,height],[1,1,1,1]])

print("Start alignment")
alg_start = timer()
images_gray = utils.BgrToGray(images)
t, t_inv, valid_id = utils.AlignEcc(images, images_gray, ref_ind, thre=0.3)
for i in range(num_img):
    corner_out = np.matmul(np.vstack([np.array(t_inv[i]),[0,0,1]]),corner)
    corner_out[0,:] = np.divide(corner_out[0,:],corner_out[2,:])
    corner_out[1,:] = np.divide(corner_out[1,:],corner_out[2,:])
    corner_out = corner_out[..., np.newaxis]
    if i == 0:
        corner_t = corner_out
    else:
        corner_t = np.append(corner_t,corner_out,2)

# print(corner_t)
alg_end = timer()
print("Full alignment: " + str(alg_end - alg_start) + "s")

images_t = utils.ApplyTransform(images, t, MOTION_MODEL)
images_t = list(images_t[i] for i in valid_id)
images = list(images[i] for i in valid_id)
ref_ind = valid_id.index(ref_ind)
num_img = len(images_t)

################# CROP #################
min_w = np.max(corner_t[0,0:2,:])
min_w = int(np.max(np.ceil(min_w),0))
min_h = np.max(corner_t[1,[0,2],:])
min_h = int(np.max(np.ceil(min_h),0))
max_w = np.min(corner_t[0,2:3,:])
max_w = int(np.floor(max_w))
max_h = np.min(corner_t[1,[1,3],:])
max_h = int(np.floor(max_h))

################# PLOT #################
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

sum_img_t, sum_img = utils.SumAlignedImage(images_t, images)

i = 0
with open(ARGS.filetxt) as f:
    for l in f:
        line = l.splitlines()[0]
        img_t = images_t[i]
        i += 1
        print("write to: ",(ARGS.folder + 'aligned/' + line))
        cv2.imwrite((ARGS.folder + 'aligned/' + line), np.uint8(255.*img_t[min_h:max_h,min_w:max_w,:]))

cv2.imwrite(os.path.join(out_sum,'aligned.jpg'), np.uint8(255.*sum_img_t))
cv2.imwrite(os.path.join(out_sum,'orig.jpg'), np.uint8(255.*sum_img))
