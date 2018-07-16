import rawpy, argparse, os
import numpy as np
from PIL import Image
# import tifffile

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="/home/xuanerzh/Downloads/compare/",
                    required=True, help="root folder that contains the images")
parser.add_argument("--new_w", default=512, type=int, help="new target width")
ARGS = parser.parse_args()
print(ARGS)

allfiles=os.listdir(ARGS.folder)
imlist=[filename for filename in allfiles if filename[-4:] in [".arw", ".ARW"]]
num_img = len(imlist)

if not os.path.exists(ARGS.folder + 'rawjpg/'):
    os.mkdir(ARGS.folder + 'rawjpg/')

if not os.path.exists(ARGS.folder + 'rawjpg_ds/'):
    os.mkdir(ARGS.folder + 'rawjpg_ds/')

for impath in sorted(imlist):
	print("Processing %s"%(ARGS.folder + impath))
	raw = rawpy.imread(ARGS.folder + impath)
	image_raw = raw.postprocess(gamma=(1,1),
		no_auto_bright=True,
		use_camera_wb=False,
		output_bps=8)
	
	image_raw = Image.fromarray(image_raw)
	w,h = image_raw.width, image_raw.height
	ratio = ARGS.new_w / w
	image_raw_ds = image_raw.resize((int(w * ratio), int(h * ratio)), Image.ANTIALIAS)

	filename = ARGS.folder + 'rawjpg/' + impath.replace("ARW","png")
	filename_ds = ARGS.folder + 'rawjpg_ds/' + impath.replace("ARW","png")
	print("write to: %s"%(filename))
	image_raw.save(filename)
	image_raw_ds.save(filename_ds)

	# tifffile.imsave(filename, np.array(image_raw))
	# tifffile.imsave(filename_ds, np.array(image_raw_ds))
