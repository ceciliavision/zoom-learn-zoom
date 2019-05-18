from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os, argparse
from PIL import Image
import numpy as np
import utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/",
                    required=True, help="root folder that contains the images")
parser.add_argument("--num", default=7, type=int, help="number of files per dir")
ARGS = parser.parse_args()
print(ARGS)

Image.MAX_IMAGE_PIXELS = None

tname = '00001.JPG'
print(ARGS.path + tname)
isrotate = utils.readOrien_pil(ARGS.path + tname) == 3
template_f = utils.readFocal_pil(ARGS.path + tname)
print("Template image has focal length: ", template_f)

if not os.path.exists(ARGS.path + "cropped/"):
    os.mkdir(ARGS.path + "cropped/")

if isrotate:
    img_rgb = Image.open(ARGS.path + tname).rotate(180)
else:
    img_rgb = Image.open(ARGS.path + tname)

img_rgb.save(ARGS.path + "cropped/00001.JPG")

i = 1
while i < ARGS.num:
    line = "%05d.JPG"%(i+1)
    img_f = utils.readFocal_pil(ARGS.path + line)
    ratio = template_f/img_f
    print("Image %s has focal length: %s "%(ARGS.path + line, img_f))
    print("Resize by ratio %s"%(ratio))

    if isrotate:
        img_rgb = Image.open(ARGS.path + line).rotate(180)
    else:
        img_rgb = Image.open(ARGS.path + line)
    
    cropped = utils.crop_fov(np.array(img_rgb), 1./ratio)
    cropped = Image.fromarray(cropped)

    img_rgb_s = cropped.resize((int(cropped.width * ratio), int(cropped.height * ratio)), Image.ANTIALIAS)
    
    print("Write to %s"%(ARGS.path + "cropped/%05d.JPG"%(1+i)))
    img_rgb_s.save(ARGS.path + "cropped/%05d.JPG"%(1+i), quality=100)
    i += 1