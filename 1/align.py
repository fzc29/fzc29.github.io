# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import skimage as sk
import skimage.io as skio
# import opencv as cv 

# name of the input file
imname = 'proj1_data/cathedral.jpg'

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

def align_ncc(ch1, ch2):
    ch1 = ch1.np.matrix.ravel()
    ch2 = ch2.np.matrix.ravel()




    return None

# b = make reference channel 

### ag = align(g, b)
### ar = align(r, b)
# create a color image (stack images on top of each other)
im_out = np.dstack([ar, ag, b])


"""
====================
Image Display 
====================
"""
# save the image
fname = '/out_path/out_fname.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()