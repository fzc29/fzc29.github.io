import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage as sk
import skimage.io as skio

def read(image):
    return mpimg.imread(image)
    
def save_image(name, image):
    im_out_uint8 = sk.img_as_ubyte(image)
    fname = '3/rectification/'+name+'.jpg'
    skio.imsave(fname, im_out_uint8)


def compute(im1_pts, im2_pts):
    """
    each im_pts = n x 2 array 
    """
    # set up the shapes 
    n = im1_pts.shape[0]

    A = np.empty((2*n, 8))
    B = im2_pts.flatten(order='c')
    B = B.reshape(-1, 1) 

    input_m = im1_pts.flatten(order='c')
    output_m = im2_pts.flatten(order='c')

    i = 0
    while i < n*2:
        x, y = input_m[i], input_m[i+1]
        a, b = output_m[i], output_m[i+1]

        x_row = np.array([x, y, 1, 0, 0, 0, -a*x, -a*y])
        y_row = np.array([0, 0, 0, x, y, 1, -b*x, -b*y])

        A[i] = x_row
        A[i+1] = y_row 

        i += 2

    result, residuals, rank, singular_values = np.linalg.lstsq(A, B, rcond=None)

    result = np.append(result, 1)
    H = result.reshape((3,3))
    print(H)
    return H

def display_points(img_path, pts):
    im = mpimg.imread(img_path)
    plt.imshow(im)
    plt.scatter(pts[:, 0], pts[:, 1], marker="o", color="green", s=25)
    plt.show()


def find_dim(im, H):
    im = mpimg.imread(im)
    
    H = np.linalg.inv(H)
    im_height, im_width = im.shape[:2]

    # 1. Compute bounding box for output image
    corners = np.array([[0, 0, 1], [im_width-1, 0, 1], [im_width-1, im_height-1, 1], [0, im_height-1, 1]]) #top left, top right, bottom right, bottom left
    
    warped_corners = (H @ corners.T).T  
    # normalize all x(width) and y(height)values  
    warped_corners[:, 0] /= warped_corners[:, 2]
    warped_corners[:, 1] /= warped_corners[:, 2]
    
    min_x = int(np.floor(np.min(warped_corners[:, 0])))
    max_x = int(np.ceil(np.max(warped_corners[:, 0])))
    min_y = int(np.floor(np.min(warped_corners[:, 1])))
    max_y = int(np.ceil(np.max(warped_corners[:, 1])))
    
    width = max_x - min_x + 1 # take off 1
    height = max_y - min_y + 1
    return width, height, min_x, min_y

def warpImageNearestNeighbor(im, H):
    # create output image size 

    width, height, min_x, min_y = find_dim(im, H)
    im = mpimg.imread(im)
    im_height, im_width = im.shape[:2]
    output = np.zeros((height, width, im.shape[2]), dtype=im.dtype)
    H_T = np.linalg.inv(H)
    # start with output image -> try to find where in input this coordinate came from 
    for y in range(height):
       for x in range(width): 
            # if outside range, keep it as 1 (black pixel) 
            warped_x = x + min_x
            warped_y = y + min_y
            curr_point = np.array([warped_x, warped_y, 1.0])
            src_point = H_T @ curr_point

            src_x = src_point[0] / src_point[2]
            src_y = src_point[1] / src_point[2]
            near_x = int(round(src_x))
            near_y = int(round(src_y))
            
            # if within, round and take closest neighbor as the pixel value 
            if 0 <= near_x < im_width and 0 <= near_y < im_height:
                # assign pixel value 
                output[y, x] = im[near_y, near_x]

    return output

def warpImageBilinear(im,H):
    width, height, min_x, min_y = find_dim(im, H)

    im = mpimg.imread(im)
    im_height, im_width = im.shape[:2]

    output = np.zeros((height, width, im.shape[2]), dtype=im.dtype)

    H_T = np.linalg.inv(H)  

    for y in range(height):
       for x in range(width):  
          
            warped_x = x + min_x
            warped_y = y + min_y
            curr_point = np.array([warped_x, warped_y, 1.0])
            src_point = H_T @ curr_point
            src_x = src_point[0] / src_point[2]
            src_y = src_point[1] / src_point[2]
          
            if not (0 <= src_x < im_width and 0 <= src_y < im_height):
                continue  # skip outside pixels

            x0 = int(np.floor(src_x))
            x1 = x0 + 1
            y0 = int(np.floor(src_y))
            y1 = y0 + 1

            x_distance = src_x - x0
            y_distance = src_y - y0

            p_tl, p_tr, p_bl, p_br = 0, 0, 0, 0            
            if 0 <= y0 < im_height and 0 <= x0 < im_width:
                p_tl = im[y0, x0]  # top-left
            if 0 <= y0 < im_height and 0 <= x1 < im_width:
                p_tr = im[y0, x1]  # top-right
            if 0 <= y1 < im_height and 0 <= x0 < im_width:
                p_bl = im[y1, x0]  # bottom-left
            if 0 <= y1 < im_height and 0 <= x1 < im_width:
                p_br = im[y1, x1]  # bottom-right

            top_left = (1-x_distance) * (1-y_distance) * p_tl
            top_right = (x_distance) * (1-y_distance) * p_tr
            bottom_left = (1-x_distance) * (y_distance) * p_bl
            bottom_right = (x_distance)*(y_distance) * p_br

            weighted_value = top_left + top_right + bottom_left + bottom_right
            output[y, x] = weighted_value
    return output 


def create_alpha_mask(img_shape):
    h, w = img_shape[:2]
    # make coordinate grid
    y, x = np.mgrid[0:h, 0:w]
    
    # distance to image center
    cx, cy = w / 2, h / 2
    dist_x = np.abs(x - cx) / (w / 2)
    dist_y = np.abs(y - cy) / (h / 2)
    
    # combine distances (could use Euclidean norm)
    dist = np.maximum(dist_x, dist_y)
    
    # invert so center=1, edge=0
    alpha = 1.0 - dist
    alpha = np.clip(alpha, 0, 1)
    
    return alpha

def blend(reference, warped, H):
    # given the warped image already 
    ref = mpimg.imread(reference)
    ref_height, ref_width = ref.shape[:2]

    # find the correct width of the total window 
    w, h, x, y = find_dim(warped, H)
    warped = warpImageNearestNeighbor(warped, H) 

    width = max(ref_width + x, w)
    height = max(ref_height + y, h)

    output = np.zeros((height, width, ref.shape[2]), dtype=ref.dtype)

    output[0:warped.shape[0], 0:warped.shape[1]] = warped

    y1 = max(0, -y)
    x1 = max(0, -x)
    y2 = min(output.shape[0], -y + ref.shape[0])
    x2 = min(output.shape[1], -x + ref.shape[1])

    ref_y1 = max(0, y)
    ref_x1 = max(0, x)
    ref_y2 = ref_y1 + (y2 - y1)
    ref_x2 = ref_x1 + (x2 - x1)

    output[y1:y2, x1:x2] = ref[ref_y1:ref_y2, ref_x1:ref_x2]

    return output 


def blend_weighted(reference, warped, H):
    ref = mpimg.imread(reference)
    ref_height, ref_width = ref.shape[:2]
    w, h, x, y = find_dim(warped, H)
    warped = warpImageNearestNeighbor(warped, H) 

    width = max(ref_width + x, w)
    height = max(ref_height + y, h)
    
    output = np.zeros((height, width, ref.shape[2]), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    warped_mask = (warped.sum(axis=2) > 0).astype(np.float32)  # summing all non-black pixels
    output[:warped.shape[0], :warped.shape[1]] += warped * warped_mask[..., None]
    weight_map[:warped.shape[0], :warped.shape[1]] += warped_mask

    # Determine bounds for reference image placement
    y1 = max(0, -y)
    x1 = max(0, -x)
    y2 = min(output.shape[0], -y + ref.shape[0])
    x2 = min(output.shape[1], -x + ref.shape[1])

    ref_y1 = max(0, y)
    ref_x1 = max(0, x)
    ref_y2 = ref_y1 + (y2 - y1)
    ref_x2 = ref_x1 + (x2 - x1)

    # Create mask for reference
    ref_crop = ref[ref_y1:ref_y2, ref_x1:ref_x2]
    ref_mask = (ref_crop.sum(axis=2) > 0).astype(np.float32)

    output[y1:y2, x1:x2] += ref_crop * ref_mask[..., None]
    weight_map[y1:y2, x1:x2] += ref_mask

    weight_map[weight_map == 0] = 1.0
    output /= weight_map[..., None]

    return output.astype(ref.dtype)




