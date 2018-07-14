import rawpy, argparse, os
import numpy as np
from PIL import Image
import tifffile

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="/home/xuanerzh/Downloads/compare/",
                    required=True, help="root folder that contains the images")
ARGS = parser.parse_args()
print(ARGS)

allfiles=os.listdir(ARGS.folder)
imlist=[filename for filename in allfiles if filename[-4:] in [".arw", ".ARW"]]
num_img = len(imlist)

if not os.path.exists(ARGS.folder + 'rawjpg/'):
    os.mkdir(ARGS.folder + 'rawjpg/')

for impath in sorted(imlist):
	print("Processing %s"%(ARGS.folder + impath))
	raw = rawpy.imread(ARGS.folder + impath)
	image_raw = raw.postprocess(gamma=(1,1),
		no_auto_bright=True,
		use_camera_wb=False,
		output_bps=16)
	filename = ARGS.folder + 'rawjpg/' + impath.replace("ARW","tiff")
	print("write to: %s"%(filename))
	tifffile.imsave(filename, image_raw)
