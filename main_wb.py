import rawpy
import os,argparse
import numpy as np

# a hacky way to compute the white balance parameters for R,G,B
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="/home/",
	required=True, help="root folder that contains the images")
parser.add_argument("--file", default='00002', type=str, help="target file path")
ARGS = parser.parse_args()
print(ARGS)

bayer = rawpy.imread(ARGS.file)
key = ARGS.file.split('.')[0]
rgb2 = bayer.postprocess(gamma=(1, 1),
	no_auto_bright=True,
	use_camera_wb=False,
	output_bps=16)

rgb8 = bayer.postprocess(gamma=(1, 1),
	no_auto_bright=True,
	use_camera_wb=True,
	output_bps=16)

scale=[np.mean(rgb8[...,0])/np.mean(rgb2[...,0]), 
	np.mean(rgb8[...,1])/np.mean(rgb2[...,1]),
	np.mean(rgb8[...,1])/np.mean(rgb2[...,1]),
	np.mean(rgb8[...,2])/np.mean(rgb2[...,2])]

tform_txt = ARGS.folder+'/wb.txt'
with open(tform_txt, 'a') as out:
    out.write("%s:"%(key) + '\n')
    out.write(" ".join(map(str,scale))+"\n")
