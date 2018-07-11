from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from PIL import Image
import numpy as np
import os, argparse
import utils as utils
import matplotlib.pyplot as plt

# python ./main_crop.py --path /home/xuanerzh/Downloads/zoom/iphone2x_both/00055/ --filetxt /home/xuanerzh/Downloads/zoom/iphone2x_both/filename.txt --ext JPG

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/xuanerzh/Downloads/zoom/",
                    required=True, help="root folder that contains the images")
parser.add_argument("--filetxt", default=None, help="file name txt file")
parser.add_argument("--num", default=7, type=int, help="number of files per dir")
parser.add_argument("--ext", default='JPG', help="file type")
ARGS = parser.parse_args()
print(ARGS)

if not os.path.exists(ARGS.path + "cropped/"):
    os.mkdir(ARGS.path + "cropped/")

tname = '00001.' + ARGS.ext
isrotate = utils.readOrien_pil(ARGS.path + tname) == 3

if isrotate:
    img_rgb = Image.open(ARGS.path + tname).rotate(180)
else:
    img_rgb = Image.open(ARGS.path + tname)

img_rgb.save(ARGS.path + "cropped/00001.png")

template_f = utils.readFocal_pil(ARGS.path + tname)
print("Template image has focal length: ", template_f)

i = 1
with open(ARGS.filetxt) as f:
    next(f) # start from 2.jpg, 1.jpg is the template by default
    for l in f:
        if i < ARGS.num:
            line = l.splitlines()[0]
            if isrotate:
                img_rgb = Image.open(ARGS.path + line).rotate(180)
            else:
                img_rgb = Image.open(ARGS.path + line)
            img_f = utils.readFocal_pil(ARGS.path + line)
            ratio = template_f/img_f
            if 'iphone' in ARGS.path:
                ratio = 52/28.
            print("Image %s has focal length: %s "%(ARGS.path + line, img_f))
            print("Resize by ratio %s"%(ratio))
            img_rgb_s = img_rgb.resize((int(img_rgb.width * ratio), int(img_rgb.height * ratio)), Image.ANTIALIAS)
            
            left = int(abs(img_rgb.width - img_rgb_s.width)/2)
            top = int(abs(img_rgb.height - img_rgb_s.height)/2)
            right = left + img_rgb.width
            bottom = top + img_rgb.height

            cropped = img_rgb_s.crop((left, top, right, bottom))
            cropped.save(ARGS.path + "cropped/%05d.png"%(1+i), quality=100)
            i += 1
