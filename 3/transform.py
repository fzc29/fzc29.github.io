import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage as sk
import skimage.io as skio
from skimage.color import rgb2gray

# Helper Functions
def read(image, gray=False):
    if gray == False:
        im = mpimg.imread(image).astype(np.float32)  # ensure float32
        if im.dtype == np.uint8 or im.max() > 1.0:   # normalize if needed
            im /= 255.0
    else:
        im = skio.imread(image)
        im = rgb2gray(im)
    return im
    
def save_image(location, image):
    im_out_uint8 = sk.img_as_ubyte(image)
    skio.imsave(location, im_out_uint8)

def plot(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Part 1 Destination = H @ Source 
# not warped -> warped 
def compute(im1_pts, im2_pts):
    # set up the shapes 
    n = im1_pts.shape[0]
    
    # create least squares 
    A = np.empty((2*n, 8))
    B = im2_pts.flatten(order='c').reshape(-1, 1)

    for i in range(n):
        x, y = im1_pts[i]
        a, b = im2_pts[i]

        A[2*i] = np.array([x, y, 1, 0, 0, 0, -a*x, -a*y])
        A[2*i + 1] = np.array([0, 0, 0, x, y, 1, -b*x, -b*y])

    result, residuals, rank, singular = np.linalg.lstsq(A, B, rcond=None)
    result = np.append(result, 1)
    H = result.reshape((3,3))
    return H

def display_points(img_path, pts, opacity=0.7, size=25):
    # im = read(img_path)
    # plt.imshow(im)
    # plt.scatter(pts[:, 0], pts[:, 1], marker="x", color="red", s=size, alpha=opacity)
    # plt.show()
    im = read(img_path)
    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.scatter(pts[:, 0], pts[:, 1], marker="x", color="red", s=size, alpha=opacity)
    plt.show()
    return fig 

def find_dim(dest, warped, H):
    imA = read(dest)
    hA, wA = imA.shape[:2]
    
    imB = read(warped)
    hB, wB = imB.shape[:2]

    inv_H = np.linalg.inv(H)
    
    cornersA = np.array([[0,0,1],[wA-1,0,1],[wA-1,hA-1,1],[0,hA-1,1]])
    cornersB = np.array([[0,0,1],[wB-1,0,1],[wB-1,hB-1,1],[0,hB-1,1]])
    warpedB = (H @ cornersB.T).T
    warpedB[:,0] /= warpedB[:,2]
    warpedB[:,1] /= warpedB[:,2]
    
    # warped_corners = (H @ cornersB.T).T  
    # warped_corners[:, 0] /= warped_corners[:, 2]
    # warped_corners[:, 1] /= warped_corners[:, 2]

    all_x = np.concatenate([cornersA[:,0], warpedB[:,0]])
    all_y = np.concatenate([cornersA[:,1], warpedB[:,1]])

    min_x = int(np.floor(all_x.min()))
    max_x = int(np.ceil(all_x.max()))
    min_y = int(np.floor(all_y.min()))
    max_y = int(np.ceil(all_y.max()))
    
    # min_x = np.min(warped_corners[:, 0])
    # max_x = np.max(warped_corners[:, 0])
    # min_y = np.min(warped_corners[:, 1])
    # max_y = np.max(warped_corners[:, 1])
    
    # width = int(np.ceil(max_x - min_x))
    # height = int(np.ceil(max_y - min_y))
    # x_off = int(np.floor(min_x))
    # y_off = int(np.floor(min_y))

    width = int(max_x - min_x)
    height = int(max_y - min_y)
    x_off = int(min_x)
    y_off = int(min_y)

    translation = np.array([[1, 0, -x_off],
                            [0, 1, -y_off],
                            [0, 0, 1]], dtype=np.float32)

    print(width, height, x_off, y_off)
    return width, height, x_off, y_off, translation @ H

def warpImageNearestNeighbor(im_path, new_H, width, height):
    # create output image size 
    # width, height, x_off, y_off, new_H = find_dim(im_path, H)
    im = read(im_path)
    im_height, im_width = im.shape[:2]

    output = np.zeros((height, width, im.shape[2]), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    H_T = np.linalg.inv(new_H)

    # start with output image -> try to find where in input this coordinate came from 
    for x in range(width):
       for y in range(height): 
            # if outside range, keep it as 1 (black pixel) 
            # warped_x, warped_y = x + x_off, y + y_off
            # curr_point = np.array([warped_x, warped_y, 1])
            # src_point = H_T @ curr_point

            curr_point = np.array([x, y, 1])
            src_point = H_T @ curr_point

            src_x = src_point[0] / src_point[2]
            src_y = src_point[1] / src_point[2]

            near_x = int(round(src_x))
            near_y = int(round(src_y))
            
            # if within bounds, round and take closest neighbor as the pixel value 
            if 0 <= near_x < im_width and 0 <= near_y < im_height:
                # assign pixel value 
                output[y, x, :] = im[near_y, near_x, :]
                mask[y, x] = 1

    return output, mask 

def warpImageBilinear(im_path, new_H, width, height):
    # width, height, x_off, y_off = find_dim(im_path, H)
    im = read(im_path)
    # print("im size: ", im.shape)
    im_height, im_width = im.shape[:2]

    output = np.zeros((height, width, im.shape[2]), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    H_T = np.linalg.inv(new_H)  

    for y in range(height):
       for x in range(width):  
            # warped_x, warped_y = x + x_off, y + y_off
            # curr_point = np.array([warped_x, warped_y, 1.0])
            # src_point = H_T @ curr_point

            curr_point = np.array([x, y, 1])
            src_point = H_T @ curr_point
          
            if src_point[2] == 0:
                continue
            src_x = src_point[0] / src_point[2]
            src_y = src_point[1] / src_point[2]
          
            if not (0 <= src_x < im_width - 1 and 0 <= src_y < im_height - 1):
                continue  # skip outside pixels

            mask[y, x] = 1

            x0 = int(np.floor(src_x)) #left
            x1 = x0 + 1 #right

            y0 = int(np.floor(src_y)) #top
            y1 = y0 + 1 #bottom

            x_distance = src_x - x0
            y_distance = src_y - y0

            p_tl = im[y0, x0]  # top-left
            p_tr = im[y0, x1]  # top-right
            p_bl = im[y1, x0]  # bottom-left
            p_br = im[y1, x1]  # bottom-right

            top = (1 - x_distance) * p_tl + x_distance * p_tr
            bottom = (1 - x_distance) * p_bl + x_distance * p_br
            weighted_value = (1 - y_distance) * top + y_distance * bottom

            # weighted_value = top_left + top_right + bottom_left + bottom_right
            output[y, x] = weighted_value
    # print("output size: ", output.shape)
    return output, mask  


def blend_n(dest, warped, mask, width, height, x_off, y_off, sigma):
    dest = read(dest)
    dest_height, dest_width = dest.shape[:2]
    dest_canvas = np.zeros((height, width, 3), dtype=np.float32)

    y_start, y_end = -y_off, -y_off + dest_height
    x_start, x_end = -x_off, -x_off + dest_width

    y_end = min(y_end, height)
    x_end = min(x_end, width)

    dest_crop = dest[:y_end - y_start, :x_end - x_start]

    dest_canvas[-y_off:-y_off+dest_height, -x_off:-x_off+dest_width] = dest_crop

    result = laplacian_pyramid_blend_2layers(dest_canvas, warped, mask)
    print("result: ", result.shape)
    plot(result)
    return result


def laplacian_pyramid_blend_2layers(imgA, imgB, mask):
    H, W = imgA.shape[:2]
    imgA = imgA.astype(np.float32)
    imgB = imgB.astype(np.float32)
    mask = mask.astype(np.float32)

    if mask.ndim == 2:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Gaussian reduce
    G2_A = cv2.pyrDown(cv2.GaussianBlur(imgA, (5,15), 2))
    G2_B = cv2.pyrDown(cv2.GaussianBlur(imgB, (5,15), 2))
    M2   = cv2.pyrDown(cv2.GaussianBlur(mask, (5,35), 2))

    # build Laplacian layer
    L1_A = imgA - cv2.pyrUp(G2_A, dstsize=(W,H))
    L1_B = imgB - cv2.pyrUp(G2_B, dstsize=(W,H))

    # Blend per level
    BL1 = mask * L1_B + (1-mask) * L1_A
    BL2 = M2 * G2_B + (1-M2) * G2_A

    # Reconstruct final blend
    BL2_up = cv2.pyrUp(BL2, dstsize=(W,H))
    blended = BL1 + BL2_up
    return np.clip(blended, 0, 1)


def make_overlap_mask(imgA, imgB, sigma=10, kernel=(51,51)):
    maskA = np.any(imgA > 0, axis=2).astype(np.uint8) 
    maskB = np.any(imgB > 0, axis=2).astype(np.uint8)

    # Intersection → overlap region
    overlap_mask = cv2.bitwise_and(maskA, maskB).astype(np.float32)  # where both have data
    seam_edges = cv2.Canny((overlap_mask*255).astype(np.uint8), 50, 150) # find edges 

    # overlap_mask_blur = cv2.GaussianBlur(overlap_mask, kernel, sigma)
    # overlap_mask_blur = np.clip(overlap_mask_blur, 0.0, 1.0)

    seam = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    seam_mask = cv2.dilate(seam_edges, seam).astype(np.float32) / 255.0

    seam_mask = cv2.GaussianBlur(seam_mask, kernel, sigma)
    seam_mask = np.clip(seam_mask, 0.0, 1.0)

    mask = np.repeat(seam_mask[:, :, np.newaxis], 3, axis=2)
    return mask


















def blend_idk(reference, warped, mask, x_off, y_off, sigma):
    # given the warped image already 
    ref = mpimg.imread(reference)
    ref_height, ref_width = ref.shape[:2]
    warped_height, warped_width = warped.shape[:2]

    left_pad = max(0, -x_off)
    top_pad = max(0, -y_off)

    # Determine final canvas
    final_w = max(ref_width + left_pad, x_off + warped_width + left_pad)
    final_h = max(ref_height + top_pad, y_off + warped_height + top_pad)

    output_ref = np.zeros((final_h, final_w, ref.shape[2]), dtype=ref.dtype)
    output_warp = np.zeros((final_h, final_w, ref.shape[2]), dtype=ref.dtype)

    ref_x_start = left_pad
    ref_y_start = top_pad
    output_ref[ref_y_start:ref_y_start+ref_height, ref_x_start:ref_x_start+ref_width] = ref

    plot(output_ref)

    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (5,15), sigma)
    mask_f = np.clip(mask_f, 0.0, 1.0)

    warp_x_start = x_off + left_pad
    warp_y_start = y_off + top_pad
    # output[warp_y_start:warp_y_start+warped_height, warp_x_start:warp_x_start+warped_width] = ref
    # ref_region = output[warp_y_start:warp_y_start+warped_height,
    #                     warp_x_start:warp_x_start+warped_width]

    # Feathered blend:
    # mask_f = 1 means warped fully visible, 0 means reference fully visible
    # blended_region = (mask_f)[..., None] * warped + (mask_f)[..., None] * ref_region

    # Place blended result in canvas
    output_warp[warp_y_start:warp_y_start+warped_height, warp_x_start:warp_x_start+warped_width] = warped
    plot(output_warp)

    return 0

def blend(reference, warped, mask, x_off, y_off, sigma):
    ref = read(reference) 
    ref_height, ref_width = ref.shape[:2]
    warped_height, warped_width = warped.shape[:2]

    left_pad = max(0, -x_off)
    top_pad = max(0, -y_off)

    final_w = max(ref_width + left_pad, x_off + warped_width + left_pad)
    final_h = max(ref_height + top_pad, y_off + warped_height + top_pad)

    output_ref = np.zeros((final_h, final_w, ref.shape[2]), dtype=np.float32)
    output_warp = np.zeros((final_h, final_w, warped.shape[2]), dtype=np.float32)
    full_mask = np.zeros((final_h, final_w), dtype=np.float32)

    output_ref[top_pad:top_pad+ref_height, left_pad:left_pad+ref_width] = ref
    output_warp[y_off+top_pad:y_off+top_pad+warped_height,
                x_off+left_pad:x_off+left_pad+warped_width] = warped
    full_mask[y_off+top_pad:y_off+top_pad+warped_height,
              x_off+left_pad:x_off+left_pad+warped_width] = mask

    full_mask = cv2.GaussianBlur(full_mask, (5,15), sigma)
    full_mask = np.clip(full_mask, 0.0, 1.0)

    blended = laplacian_pyramid_blend_2layers(output_ref, output_warp, full_mask)
    return blended






