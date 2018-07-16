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
# parser.add_argument("--filetxt", default=None, help="file name txt file")
parser.add_argument("--subfolder", default='rawjpg/', help="file extension")
parser.add_argument("--num", default=7, type=int, help="number of files per dir")
parser.add_argument("--ext", default='png', help="file extension")
parser.add_argument("--type", default='raw', help="file type")
ARGS = parser.parse_args()
print(ARGS)

Image.MAX_IMAGE_PIXELS = None

tname = '00001.JPG'
isrotate = utils.readOrien_pil(ARGS.path + tname) == 3
template_f = utils.readFocal_pil(ARGS.path + tname)
print("Template image has focal length: ", template_f)

if ARGS.type == 'raw':
    path = ARGS.path + ARGS.subfolder
else:
    path = ARGS.path

if not os.path.exists(path + "cropped/"):
    os.mkdir(path + "cropped/")


tname_raw = tname.replace('JPG', ARGS.ext)

if isrotate:
    img_rgb = Image.open(path + tname_raw).rotate(180)
else:
    img_rgb = Image.open(path + tname_raw)

img_rgb.save(path + "cropped/00001.png")

i = 1
while i < ARGS.num:
    line = "%05d.JPG"%(i+1)
    img_f = utils.readFocal_pil(ARGS.path + line)
    ratio = template_f/img_f
    if 'iphone' in ARGS.path:
        ratio = 52/28.
    print("Image %s has focal length: %s "%(ARGS.path + line, img_f))
    print("Resize by ratio %s"%(ratio))

    if ARGS.type == 'raw':
        line = line.replace('JPG', ARGS.ext)
    if isrotate:
        img_rgb = Image.open(path + line).rotate(180)
    else:
        img_rgb = Image.open(path + line)
    
    cropped = utils.crop_fov(np.array(img_rgb), 1./ratio)
    cropped = Image.fromarray(cropped)

    img_rgb_s = cropped.resize((int(cropped.width * ratio), int(cropped.height * ratio)), Image.ANTIALIAS)
    
    # left = int(abs(img_rgb.width - img_rgb_s.width)/2)
    # top = int(abs(img_rgb.height - img_rgb_s.height)/2)
    # right = left + img_rgb.width
    # bottom = top + img_rgb.height
    # cropped = img_rgb_s.crop((left, top, right, bottom))
    
    img_rgb_s.save(path + "cropped/%05d.%s"%(1+i,ARGS.ext), quality=100)
    i += 1