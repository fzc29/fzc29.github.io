# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2


# name of the input file
imname = '1/proj1_data/monastery.jpg'

# read in the image
im = skio.imread(imname)

""" 
====================
Efficiency Measures
====================
"""
# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)

"""
====================
Image Separation
====================
"""
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]


# clean up picture frames (cropping)
row, col = b.shape
b_cropped = b[int(col*0.1): int(col - 0.1*col), int(0.1*row):int(row - 0.1*row)] 
g_cropped = g[int(col*0.1): int(col - 0.1*col), int(0.1*row):int(row - 0.1*row)] 
r_cropped = r[int(col*0.1): int(col - 0.1*col), int(0.1*row):int(row - 0.1*row)] 

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
    denominator = np.sqrt(np.sum(I**2) * np.sum(P**2))
    return numerator / denominator

def align_ncc(base, match, x_shift, y_shift):
    """
    using base as reference, match match to base 
    """
    best_ncc = -1
    # nested loop for rolling x and y 
    for x in range(x_shift-15, x_shift+15):
        for y in range(y_shift-15, y_shift+15):
            temp = np.roll(np.roll(match, y, axis=0), x, axis=1)
            cur_ncc = ncc(base, temp)
            if cur_ncc > best_ncc:
                x_shift, y_shift = x, y
                best_ncc = cur_ncc
    return (x_shift, y_shift)

"""
====================
Pyramid
====================
"""

def pyramid_align(base, match, level, edge=False):
    base_sizes = [base]
    match_sizes = [match]
    i = 0

    while i < level:
        minimized_base = cv2.resize(base_sizes[0], dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        minimized_match = cv2.resize(match_sizes[0], dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # base_sizes.append(minimized_base)
        base_sizes.insert(0, minimized_base)
        # match_sizes.append(minimized_match)
        match_sizes.insert(0, minimized_match)
        i += 1
 
    x_shift, y_shift = 0, 0

    for i in range(0, level):
        if edge == True:
            cur_base = sobel(base_sizes[i])
            cur_match = sobel(match_sizes[i])
        else: 
            cur_base = base_sizes[i]
            cur_match = match_sizes[i]
        shift = align_ncc(cur_base, cur_match, x_shift, y_shift)
        x_shift, y_shift = shift[0] * 2, shift[1] * 2
        print(x_shift, y_shift)
    
    return (x_shift, y_shift)

def sobel(matrix):
    G_x = np.array([[-1, 0, 1], 
                     [-2, 0, 2], 
                     [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], 
                     [0, 0, 0], 
                     [1, 2, 1]])
    
    # convolve Image with G_x and G_y
    
    I_x = cv2.filter2D(src=matrix, ddepth=-1, kernel=G_x)
    
    I_y = cv2.filter2D(src=matrix, ddepth=-1, kernel=G_y)

    out_mat = np.sqrt((I_x**2) + (I_y**2))
    return out_mat


# # b = reference channel 
# #ag = align_ncc(b_cropped, g_cropped, 0, 0)
# ag = pyramid_align(b_cropped, g_cropped, 5)
# print(ag)
# g_shifted = np.roll(np.roll(g, ag[1], axis=0), ag[0], axis=1)

# #ar = align_ncc(b_cropped, r_cropped, 0, 0)
# ar = pyramid_align(b_cropped, r_cropped, 5)
# print(ar)
# r_shifted = np.roll(np.roll(r, ar[1], axis=0), ar[0], axis=1)

# # create a color image (stack images on top of each other)
# im_out = np.dstack([r_shifted, g_shifted, b])

"""
====================
No Alignment 
====================
"""

im_out = np.dstack([r, g, b])

"""
====================
Image Display 
====================
"""
# save the image
im_out_uint8 = sk.img_as_ubyte(im_out)
fname = '1/final_result/no_align_monastery.jpg'
skio.imsave(fname, im_out_uint8)

# display the image
skio.imshow(im_out)
skio.show()


