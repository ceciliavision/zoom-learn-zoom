from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os, cv2, argparse
import utils as utils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/xuanerzh/Downloads/",
                    required=True, help="root folder that contains the images")
parser.add_argument("--outname", type=str, default='fusion.jpg', required=False, help="Output images name")
parser.add_argument("--filetxt", default=None, help="file name txt file")
ARGS = parser.parse_args()
print(ARGS)

MOTION_MODEL = "ECC"

tname = 'DSC03312.JPG'
img_rgb = cv2.imread(ARGS.path + tname)
cv2.imwrite(ARGS.path + "00001.jpg", img_rgb)

template_f = utils.readFocal(ARGS.path + tname)
print(img_rgb.shape, "image has focal length: ", template_f)
img_template = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
w, h = img_template.shape[::-1]
corner=np.array([[0,0,w,w],[0,h,0,h],[1,1,1,1]])

i = 1
with open(ARGS.filetxt) as f:
    for l in f:
        line = l.splitlines()[0]
        img_rgb = cv2.imread(ARGS.path + line)
        print(ARGS.path + line)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_f = utils.readFocal(ARGS.path + line)
        ratio = template_f/img_f
        print("image %s has focal length: "%(line), img_f, ratio)
        img_rgb = cv2.resize(img_rgb, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        img_gray = cv2.resize(img_gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        res = cv2.matchTemplate(img_gray,img_template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        print("locate match: ", top_left, bottom_right)

        cv2.imwrite((ARGS.path + "%05d.jpg"%(1+i)), 
            img_rgb[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0],:])
        i += 1

        # plt.subplot(121),plt.imshow(img_template,cmap = 'gray')
        # plt.subplot(122),plt.imshow(img_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]],cmap = 'gray')
        # plt.show()
