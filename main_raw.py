import rawpy, argparse, os
import numpy as np
from PIL import Image
# import tifffile

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="/home/xuanerzh/Downloads/zoom/dslr_10x_both/00065/",
                    required=True, help="root folder that contains the images")
parser.add_argument("--new_w", default=512, type=int, help="new target width")
ARGS = parser.parse_args()
print(ARGS)

allfiles=os.listdir(ARGS.folder)
imlist=[filename for filename in allfiles if filename[-4:] in [".arw", ".ARW"]]
num_img = len(imlist)

p_folder = os.path.basename(os.path.dirname(os.path.dirname(ARGS.folder)))
save_folder = ARGS.folder.replace(p_folder, p_folder+"_process")

if not os.path.exists(save_folder + 'rawpng/'):
    os.makedirs(save_folder + 'rawpng/')

if not os.path.exists(save_folder + 'rawpng_ds/'):
    os.makedirs(save_folder + 'rawpng_ds/')

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

	p_folder = os.path.basename(os.path.dirname(ARGS.folder))
	filename = save_folder + 'rawpng/' + impath.replace("ARW","png")
	filename_ds = save_folder + 'rawpng_ds/' + impath.replace("ARW","png")
	print("write to: %s"%(filename))
	image_raw.save(filename)
	image_raw_ds.save(filename_ds)

	# tifffile.imsave(filename, np.array(image_raw))
	# tifffile.imsave(filename_ds, np.array(image_raw_ds))
