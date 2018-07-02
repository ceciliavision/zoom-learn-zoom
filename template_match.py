from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from PIL import Image
import numpy as np
import os, argparse
import utils as utils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/xuanerzh/Downloads/zoom/",
                    required=True, help="root folder that contains the images")
parser.add_argument("--filetxt", default=None, help="file name txt file")
ARGS = parser.parse_args()
print(ARGS)

if not os.path.exists(ARGS.path + "cropped/"):
    os.mkdir(ARGS.path + "cropped/")

tname = '00001.jpg'
img_rgb = Image.open(ARGS.path + tname)
img_rgb.save(ARGS.path + "cropped/00001.png")

template_f = utils.readFocal_pil(ARGS.path + tname)
print("image has focal length: ", template_f)
# img_template = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# w, h = img_template.shape[::-1]
# corner=np.array([[0,0,w,w],[0,h,0,h],[1,1,1,1]])

i = 1
with open(ARGS.filetxt) as f:
    next(f) # start from 2.jpg, 1.jpg is the template by default
    for l in f:
        line = l.splitlines()[0]
        img_rgb = Image.open(ARGS.path + line)
        print(ARGS.path + line)
        # img_gray = img_rgb.convert('LA')
        img_f = utils.readFocal_pil(ARGS.path + line)
        ratio = template_f/img_f
        print("image %s has focal length: "%(line), img_f, ratio)
        img_rgb_s = img_rgb.resize((int(img_rgb.width * ratio), int(img_rgb.height * ratio)))
        # img_gray_s = img_gray.resize(img_gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        
        left = int(abs(img_rgb.width - img_rgb_s.width)/2)
        top = int(abs(img_rgb.height - img_rgb_s.height)/2)
        right = left + img_rgb.width
        bottom = top + img_rgb.height

        cropped = img_rgb_s.crop((left, top, right, bottom))
        cropped.save(ARGS.path + "cropped/%05d.png"%(1+i))
        i += 1

        # res = cv2.croppedTemplate(img_gray,img_template,cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # top_left = max_loc
        # bottom_right = (top_left[0] + w, top_left[1] + h)
        # print("locate cropped: ", top_left, bottom_right)

        # cv2.imwrite((ARGS.path + "cropped/%05d.jpg"%(1+i)), 
        #     img_rgb[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0],:])
        # i += 1
