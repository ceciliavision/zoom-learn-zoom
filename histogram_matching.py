import argparse, os
import numpy as np
from scipy.misc import imread, imsave


parser = argparse.ArgumentParser()
parser.add_argument('--template_dir', required=True) # Low-res images
parser.add_argument('--source_dir', required=True)   # Outputs from CNN
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()


def main():
  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  for fn in os.listdir(args.template_dir):
    template_img = imread(os.path.join(args.template_dir, fn))
    source_img = imread(os.path.join(args.source_dir, fn))

    if template_img.ndim == 2:
      template_img = template_img[:, :, None][:, :, [0, 0, 0]]
      
    out_img = np.zeros_like(source_img)
    for c in xrange(3):
      out_img[:, :, c] = hist_match(source_img[:, :, c], template_img[:, :, c])

    imsave(os.path.join(args.output_dir, fn), out_img)


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.

    Code adapted from
    http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
  
  
if __name__ == '__main__':
  main()
