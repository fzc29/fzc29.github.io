# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import skimage as sk
import skimage.io as skio
# import opencv as cv 

# name of the input file
imname = '1/proj1_data/cathedral.jpg'

# read in the image
im = skio.imread(imname)

""" 
====================
Efficiency Measures
====================
"""
# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)

# compute at lower resolution (zoom out?) 
    
"""
====================
Image Separation
====================
"""
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# clean up edges 


"""
====================
Image Alignment
====================
"""

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def ncc(base, match):
    # calculating NCC value for 2 matrixes 
    I = base - np.mean(base)
    P = match - np.mean(match)
    numerator = np.sum(I*P)
    denominator = np.sqrt(np.sum(I**2)) * np.sqrt(np.sum(P**2))
    return numerator / denominator

def align_ncc(base, match):
    """
    using base as reference, match match to base 
    """
    x_shift, y_shift = 0, 0
    best_ncc = 0 
    row, col = base.shape 
    # nested loop for rolling x and y 
    for x in range(0, row//2):
        for y in range(0, col//2):
            temp = np.roll(np.roll(match, x, axis=0), y, axis=1)
            cur_ncc = ncc(base, temp)
            if cur_ncc > best_ncc:
                x_shift, y_shift = x, y
                best_ncc = cur_ncc

    return (x_shift, y_shift)

# b = reference channel 
ag = align_ncc(b, g)
g_shifted = np.roll(np.roll(g, ag[0], axis=0), ag[1], axis=1)

ar = align_ncc(b, r)
r_shifted = np.roll(np.roll(r, ar[0], axis=0), ar[1], axis=1)

# create a color image (stack images on top of each other)
im_out = np.dstack([r_shifted, g_shifted, b])

"""
====================
Image Display 
====================
"""
# save the image
fname = '1/result_data/cathedral_out.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()