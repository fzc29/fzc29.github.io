import numpy as np 
import cv2
import scipy as sc
import math
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage as sk

"""
====================
1.1 Convolutions from Scratch!
====================
"""
def convolve_4loop(image, kernel):
    """
    Convolution with 4 loops 

    Input: 
    Image = 2d numpy array (gray scale)
    Kernel = 2d nump array (assume all odd dimensions)

    Output:
    Convoluted kernel with image using "same" mode padding of 0's 
    """
     
    m_rows, m_cols = image.shape
    k_rows, k_cols = kernel.shape

    padding_td = (k_rows - 1) // 2
    padding_lr = (k_cols - 1) // 2

    output = np.zeros(image.shape, dtype=np.float64)

    # flip kernel twice to reverse it 
    kernel = np.flip(kernel)

    # pad image (same)
    image = np.pad(image, ((padding_td, padding_td), (padding_lr, padding_lr)), mode='constant', constant_values=0)

    # convolve
    for i in range(m_rows):
        for j in range(m_cols):
            pixel = 0
            for row in range(k_rows):
                for col in range(k_cols): 
                    pixel += image[i+row][j+col] * kernel[row][col]
            output[i][j] = pixel                 
    
    return output


def convolve_2loop(image, kernel):
    """
    Same as above with only 2 loops 
    """
    m_rows, m_cols = image.shape
    k_rows, k_cols = kernel.shape

    padding_td = (k_rows - 1) // 2
    padding_lr = (k_cols - 1) // 2

    output = np.zeros(image.shape, dtype=np.float64)

    # flip kernel twice to reverse it 
    kernel = np.flip(kernel)

    # pad image (same)
    image = np.pad(image, ((padding_td, padding_td), (padding_lr, padding_lr)), mode='constant', constant_values=0)

    # using np.sum -> already vectorized function 
    for i in range(m_rows):
        for j in range(m_cols):
            pixel = np.sum(kernel * image[i: i + k_rows, j: j + k_cols])
            output[i][j] = pixel
    return output 


"""
====================
Part 1.2: Finite Difference Operator
====================
"""

def theshold(im, num):
    return (im > num)

"""
====================
Part 1.3: Derivative of Gaussian (DoG) Filter
====================
"""

def gauss(size, sigma):
    g = cv2.getGaussianKernel(size, sigma)
    # gT = g.T
    # out = np.matmul(g, gT)
    return g @ g.T

def normalize_filter(fil):
    fil = fil - fil.min()
    fil = fil / fil.max()
    return (fil * 255).astype(np.uint8)

    
"""
====================
Part 2.1: Fun with Frequencies!
====================
"""

# f * ((1+alpha)e - alpha g)
def sharpen(alpha, g):
    e = np.zeros_like(g)
    h, w = g.shape
    e[h // 2, w // 2] = 1.0
    f = ((1 + alpha)*e) - (alpha * g)
    return f

"""
====================
Part 2.2: Hybrid Images 
====================
"""
def hybrid_image(im1, im2, s1, s2):
    # low pass filter using standard Gaussian 
    g1 = gauss(15,s1) 
    low = sc.signal.convolve2d(im1, g1, mode='same', boundary='fill', fillvalue=0)
    
    # high pass filter -> subtracting Gaussian filtered image from original image 
    g2 = gauss(15,s2)
    high = im2 - sc.signal.convolve2d(im2, g2, mode='same', boundary='fill', fillvalue=0)

    # add image
    out = low + 1.2*high
    return out

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        # im1 = sktr.rescale(im1, dscale)
        im1 = sktr.rescale(im1, dscale, channel_axis=2, anti_aliasing=True)
    else:
        # im2 = sktr.rescale(im2, 1./dscale)
        im2 = sktr.rescale(im2, 1.0/dscale, channel_axis=2, anti_aliasing=True)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    # assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


if __name__ == "__main__":
    # 1. load the image
    # 2. align the two images by calling align_images
    # Now you are ready to write your own code for creating hybrid images!
    pass

def graph(im, title):
    im = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))))
    plt.imshow(im, cmap='gray')
    plt.title("Log Magnitude Spectrum: "+ title, pad=20) 
    plt.xlabel("Horizontal Frequencies", labelpad=15)              
    plt.ylabel("Vertical Frequencies", labelpad=15)  
    plt.savefig('2/hybrid_fft/'+title+'.png', dpi=300)                
    plt.show()
    

def gray(image):
    if image.ndim == 3: 
        image = np.mean(image, axis=2)
    return image

"""
====================
Part 2.3: Gaussian + Laplacian Stacks
====================
"""

import time

def gauss_stack(image, sigma):
    stack = [image]
    
    for i in range(5):
        g = gauss(sigma+1, sigma)

        start = time.time()

        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[..., c] = sc.signal.convolve2d(stack[0][..., c], g, mode='same', boundary='symm')
        sigma += 1
        stack.append(blurred)

        end = time.time()
        print(str(i) + 'th image added' + ' taking ' + str(end-start) + " seconds")

    return stack

def gauss_mask(mask, sigma):
    stack = [mask]
    for _ in range(5):
        g = gauss(sigma*6 + 1, sigma)
        blurred = sc.signal.convolve2d(stack[0], g, mode='same', boundary='symm')
        sigma += 5
        stack.append(blurred)
    return stack


def laplacian_stack(gauss_stack):
    stack = []
    for i in range(len(gauss_stack)-1):
        lap = gauss_stack[i] - gauss_stack[i + 1]
        stack.append(lap)
    stack.append(gauss_stack[-1])
    return stack

def lalacianp(gauss):
    stack = []
    for i in range(4):
        start = time.time()

        lap = gauss[i] - gauss[i + 1]
        stack.append(lap)

        end = time.time()
        print(str(i) + 'th lap image added' + 'taking' + str(end-start) + "seconds")
    stack.append(gauss[-1])
    return stack

def create_mask(image, sigma):
    rows, cols = image.shape[:2]  # ignore channels
    mask = np.ones((rows, cols), dtype=np.float32)
    #left half = 1, right half = 0
    mask[:, :cols // 2] = 0

    g = gauss(sigma*6 + 1, sigma)
    mask = sc.signal.convolve2d(mask, g, mode='same', boundary='symm')
    mask = mask / mask.max()
    return mask 

def create_binary_mask(image):
    rows, cols = image.shape[:2]
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[:, :cols // 2] = 0
    return mask

def blend(im1, im2, mask):
    if im1.ndim == 3:
        mask = mask[..., np.newaxis]
    result = im1 * mask + im2 * (1-mask)
    return result

def blend_one(im1, mask):
    if im1.ndim == 3:
        mask = mask[..., np.newaxis]
    result = im1 * mask
    return result

def blend_two(im2, mask):
    if im2.ndim == 3:
        mask = mask[..., np.newaxis]
    result = im2 * (1 - mask)
    return result

def combine(laplacian_stack):
    image = laplacian_stack[-1].copy()
    for lap in reversed(laplacian_stack[:-1]):
        image += lap
    return image

"""
====================
Part 2.4: Multi-resolution Blending
====================
""" 

def create_2_mask(image):
    rows, cols = image.shape[:2]
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[: (3*rows)// 5, :] = 0
    return mask