from convolution import *
from skimage import io, color
import skimage as sk
import scipy as sc
import time
from skimage import exposure

"""
====================
Part 1
====================
"""

D_x = np.array([[1, 0, -1]])
D_y = np.array([[1], [0], [-1]])

size = 9
box = np.ones((size, size), dtype=float) / (size * size)


def part_one_one():
    imname = '2/images/me.jpg'
    im = io.imread(imname)
    im = color.rgb2gray(im)
    im = sk.img_as_float(im)
    filters = {"D_x": D_x, "D_y": D_y, "box": box}

    for name, fil in filters.items():
        start_time = time.time()
        sc.signal.convolve2d(im, fil, mode='same', boundary='fill', fillvalue=0)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Sc {name}: {elapsed:.6f} seconds")
        
# part_one_one()

def part_one_two_one():
    imname = '2/images/cameraman.jpg'
    im = io.imread(imname)
    im = im[:, :, :3]
    im = color.rgb2gray(im)
    im = sk.img_as_float(im)
    
    filters = {"Dx": D_x, "Dy": D_y, "box": box}
    for name, fil in filters.items():
        start_time = time.time()
        out = convolve_4loop(im, fil)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"4Loop Cam {name}: {elapsed:.6f} seconds") 

        fname = '2/out_images/cameraman'+name+'.jpg'
        io.imsave(fname, out)

# part_one_two_one()

def part_one_two_two():
    imname = '2/images/cameraman.jpg'
    im = io.imread(imname)
    im = im[:, :, :3]
    im = color.rgb2gray(im)
    im = sk.img_as_float(im)

    I_x = convolve_4loop(im, D_x)
    I_y = convolve_4loop(im, D_y)
    G = np.sqrt(I_x**2 + I_y**2)

    fname = '2/out_images/cameramanGradient.jpg'
    io.imsave(fname, G)

    return None

# part_one_two_two()

def part_one_two_three():
    imname = '2/images/cameraman.jpg'
    im = io.imread(imname)
    im = im[:, :, :3]
    im = color.rgb2gray(im)
    im = sk.img_as_float(im)

    I_x = convolve_4loop(im, D_x)
    I_y = convolve_4loop(im, D_y)
    G = np.sqrt(I_x**2 + I_y**2)

    for i in [0.01, 0.03, 0.05, 0.1, 0.13, 0.15, 0.17, 0.20, 0.25, 0.30, 0.40, 0.55]:
        edge = theshold(G, i)
        fname = '2/camera_threshold/cameraman_'+str(i)+'.jpg'
        io.imsave(fname, edge)

# part_one_two_three()

def part_one_three_one():
    imname = '2/images/me.jpg'
    im = io.imread(imname)
    im = color.rgb2gray(im)
    im = sk.img_as_float(im)

    g = gauss(37, 6)

    out = sc.signal.convolve2d(im, g, mode='same', boundary='fill', fillvalue=0)

    fname = '2/dog/meG(37_6).jpg'
    io.imsave(fname, out)

# part_one_three_one()

def part_one_three_two():
    g = gauss(5, 3)
    dx_g = sc.signal.convolve2d(g, D_x, mode='same', boundary='fill', fillvalue=0)
    dy_g = sc.signal.convolve2d(g, D_y, mode='same', boundary='fill', fillvalue=0)

    # dx_g = normalize_filter(dx_g)
    # dy_g = normalize_filter(dy_g)

    # fname = '2/dog/dxg.jpg'
    # io.imsave(fname, dx_g)
    # fname = '2/dog/dyg.jpg'
    # io.imsave(fname, dy_g)

    imname = '2/images/cameraman.jpg'
    im = io.imread(imname)
    im = im[:, :, :3]
    im = color.rgb2gray(im)
    im = sk.img_as_float(im)

    cam_x = sc.signal.convolve2d(im, dx_g, mode='same', boundary='fill', fillvalue=0)
    cam_y = sc.signal.convolve2d(im, dy_g, mode='same', boundary='fill', fillvalue=0)


    # fname = '2/dog/cam_dxg.jpg'
    # io.imsave(fname, cam_x)
    # fname = '2/dog/cam_dyg.jpg'
    # io.imsave(fname, cam_y)

    output = np.sqrt(cam_x**2 + cam_y**2)
    fname = '2/dog/cam_dxyg_gradient.jpg'
    io.imsave(fname, output)

# part_one_three_two()


# out = convolve_4loop(im, box) 
# out = sc.signal.convolve2d(im, D_x, mode='same', boundary='fill', fillvalue=0)
# # im_out = np.clip(out, 0, 1)
# # im_out = sk.img_as_ubyte(im_out)
# fname = '2/out_images/meD_xSC.jpg'
# io.imsave(fname, out) 

def convolve_then_d():
    g = gauss(5, 3)

    imname = '2/images/cameraman.jpg'
    im = io.imread(imname)
    im = im[:, :, :3]
    im = color.rgb2gray(im)
    im = sk.img_as_float(im)

    gaussian = sc.signal.convolve2d(im, g, mode='same', boundary='fill', fillvalue=0)
    fname = '2/dog/cam_convolved_first.jpg'
    io.imsave(fname, gaussian)

    # apply dx dy
    dx_g = sc.signal.convolve2d(gaussian, D_x, mode='same', boundary='fill', fillvalue=0)
    dy_g = sc.signal.convolve2d(gaussian, D_y, mode='same', boundary='fill', fillvalue=0)
    
    output = np.sqrt(dx_g**2 + dy_g**2)
    fname = '2/dog/cam_dxdy_after.jpg'
    io.imsave(fname, output)

# convolve_then_d()

def combine_filter_then():
    g = gauss(4, 3)
    g1 = sc.signal.convolve2d(g, D_x, mode='same', boundary='fill', fillvalue=0)
    g2 = sc.signal.convolve2d(g, D_y, mode='same', boundary='fill', fillvalue=0)


    fname = '2/dog/cam_filters_g1.jpg'
    io.imsave(fname, g1)
    fname = '2/dog/cam_filters_g2.jpg'
    io.imsave(fname, g2)

    imname = '2/images/cameraman.jpg'
    im = io.imread(imname)
    im = im[:, :, :3]
    im = color.rgb2gray(im)
    im = sk.img_as_float(im)

    gaussian1 = sc.signal.convolve2d(im, g1, mode='same', boundary='fill', fillvalue=0)
    gaussian2 = sc.signal.convolve2d(im, g2, mode='same', boundary='fill', fillvalue=0)
    output = np.sqrt(gaussian1**2 + gaussian2**2)
    fname = '2/dog/cam_combined_then.jpg'
    io.imsave(fname, output)
    

# combine_filter_then()

"""
====================
Part 2
====================
"""

def part_two_one_taj():
    imname = '2/sharpen/taj.jpg'
    im = io.imread(imname)
    im = sk.img_as_float(im)

    g = gauss(9, 2) 
    m = sharpen(2, g)

    out = np.zeros_like(im, dtype=np.float64)

    for i in range(3):
        plane = im[..., i]
        result = sc.signal.convolve2d(plane, m, mode='same', boundary='fill', fillvalue=0)
        out[..., i] = result 
    
    
    fname = '2/sharpen/taj_2.jpg'
    io.imsave(fname, np.clip(out, 0, 1))

# part_two_one_taj()

def part_two_one_my():

    g = gauss(9, 2) 
    m = sharpen(3, g)

    imname = '2/my_sharpen/dessert.jpg'
    im = io.imread(imname)
    im = sk.img_as_float(im)

    out = np.zeros_like(im, dtype=np.float64)
    for i in range(3):
        plane = im[..., i]
        result = sc.signal.convolve2d(plane, m, mode='same', boundary='fill', fillvalue=0)
        out[..., i] = result 

    fname = '2/my_sharpen/dessert_3.jpg'
    io.imsave(fname, np.clip(out, 0, 1))

# part_two_one_my()

def part_two_one_blurthensharpen():
    imname = '2/my_sharpen/shell.jpg'
    im = io.imread(imname)
    im = sk.img_as_float(im)

    blur = np.zeros_like(im, dtype=np.float64)
    for i in range(3):
        plane = im[..., i]
        result = sc.signal.convolve2d(plane, box, mode='same', boundary='fill', fillvalue=0)
        blur[..., i] = result 
    
    fname = '2/my_sharpen/shell_blur.jpg'
    io.imsave(fname, np.clip(blur, 0, 1))

    g = gauss(9, 2) 
    m = sharpen(2, g)

    out = np.zeros_like(im, dtype=np.float64)
    for i in range(3):
        plane = im[..., i]
        result = sc.signal.convolve2d(plane, m, mode='same', boundary='fill', fillvalue=0)
        out[..., i] = result 
    
    fname = '2/my_sharpen/shell_blurthensharp.jpg'
    io.imsave(fname, np.clip(out, 0, 1))

# part_two_one_blurthensharpen() 

def part_two_hybrid():
    # # high sf
    # high_sf = plt.imread('./bro.jpg')/255.
    # # low sf
    # low_sf = plt.imread('./pomeranian.jpg')/255

    # practice:
    # im1 = plt.imread('2/hybrid_python/DerekPicture.jpg')/255.
    # im2 = plt.imread('2/hybrid_python/nutmeg.jpg')/255

    # pom bro
    # im1 = plt.imread('2/hybrid/bro.jpg')/255.
    # im2 = plt.imread('2/hybrid/pomeranian.jpg')/225

    im2 = plt.imread('2/hybrid/3-giyu.jpg')/255
    im1 = plt.imread('2/hybrid/3-jack.jpg')/225

    # im1 = plt.imread('2/hybrid/pagoda.jpg')/255
    # im2 = plt.imread('2/hybrid/empire.jpg')/225
    
    im1_aligned, im2_aligned = align_images(im1, im2)

    im1_aligned = color.rgb2gray(im1_aligned)
    im1_aligned = sk.img_as_float(im1_aligned)

    # im2_aligned = exposure.rescale_intensity(im2_aligned, in_range='image', out_range=(0,1)) # for pombro
    # im2_aligned = exposure.adjust_gamma(im2_aligned, gamma=2) 
    im2_aligned = color.rgb2gray(im2_aligned)
    im2_aligned = sk.img_as_float(im2_aligned)

    sigma1 = 15 # for low_pass blur
    sigma2 = 7 # for high_pass subtraction 
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

    plt.imshow(hybrid)
    plt.show()

    fname = '2/hybrid/jayu.jpg'
    io.imsave(fname, np.clip(hybrid, 0, 1))

# part_two_hybrid()

def part_two_hybrid_analysis():
    # using the building example 

    # image 1 input 
    im1 = plt.imread('2/hybrid/1-pagoda.jpg')
    im1 = gray(im1)
    graph(im1, "Input_Pagoda")

    # image 2 input 
    im2 = plt.imread('2/hybrid/1-pagoda.jpg')
    im2 = gray(im1)
    graph(im2, "Input_Empire")

    # image 1 with low pass 
    g1 = gauss(15, 15) 
    low = sc.signal.convolve2d(im1, g1, mode='same', boundary='fill', fillvalue=0)
    low = gray(low)
    graph(low, "Filtered_Pagoda")


    # image 2 with high pass
    g2 = gauss(15,15) 
    high = im2 - sc.signal.convolve2d(im2, g2, mode='same', boundary='fill', fillvalue=0)
    high = gray(high)
    graph(high, "Filtered_Empire")

    out = low + 1.2*high
    out = gray(out)
    graph(out, "Hybrid_Image")

# part_two_hybrid_analysis()

def part_two_hybrid_2_analysis():
    # using the giyu example 

    # image 1 input 
    im1 = plt.imread('2/hybrid/3-giyu.jpg')
    im1 = gray(im1)
    graph(im1, "Giyu")

    # image 2 input 
    im2 = plt.imread('2/hybrid/3-jack.jpg')
    im2 = gray(im1)
    graph(im2, "Jack")

    # image 1 with low pass 
    g1 = gauss(15, 15) 
    low = sc.signal.convolve2d(im1, g1, mode='same', boundary='fill', fillvalue=0)
    low = gray(low)
    graph(low, "Filtered_Giyu")


    # image 2 with high pass
    g2 = gauss(15,7) 
    high = im2 - sc.signal.convolve2d(im2, g2, mode='same', boundary='fill', fillvalue=0)
    high = gray(high)
    graph(high, "Filtered_Jack")

    out = low + 1.2*high
    out = gray(out)
    graph(out, "Hybrid_Jiyu")

# part_two_hybrid_2_analysis()

sigma1 = 15
sigma2 = 7

f_c1 = np.sqrt(np.log(2)) / (2 * np.pi * sigma1)
f_c2 = np.sqrt(np.log(2)) / (2 * np.pi * sigma2)

print("Low-pass cutoff frequency fc1:", f_c1, "cycles per pixel")
print("High-pass cutoff frequency fc2:", f_c2, "cycles per pixel")

def stacks():
    right = '2/stack/apple.jpg'
    left = '2/stack/orange.jpg'

    #load pictures 
    left = io.imread(left)/255
    right = io.imread(right)/255
   
    #get mask (2D)
    mask =  create_mask(left, 40) 
  
    #laplacian of both images
    left_lap = laplacian_stack(left, 40)

    right_lap = laplacian_stack(right, 40)

    
    # combine 
    total = []
    for apple, orange in zip(left_lap, right_lap):
        result = blend(apple, orange, mask)
        total.append(result)



    final = combine(total)

    fname = '2/stack/oraple2.jpg'
    io.imsave(fname, np.clip(final, 0, 0.1))

# stacks()

def save(thing, name):
    i = 0 
    for i in range(len(thing)):
        filename = f"2/oraple/final/{name}_{i}.png"
        io.imsave(filename, (thing[i] * 255).astype(np.uint8))

# left = '2/stack/apple.jpg'
# right = '2/stack/orange.jpg'
# left = io.imread(left)/255
# right = io.imread(right)/255

# left_stack = gauss_stack(left, 1)
# right_stack = gauss_stack(right, 1)

# left_lap = laplacian_stack(left_stack)

# print("finished 1")
# right_lap = laplacian_stack(right_stack)

# print("finished 2")

# mask =  create_binary_mask(left) 
# g = gauss(100, 40)
# blurred = sc.signal.convolve2d(mask, g, mode='same', boundary='symm')

# mask_stack = gauss_mask(mask, 10)
# left_blur = blend_one(left, 1-blurred)
# filename = f"2/oraple/final/left_blur.png"
# io.imsave(filename, (left_blur * 255).astype(np.uint8))

# right_blur = blend_two(right, 1-blurred)
# filename = f"2/oraple/final/right_blur.png"
# io.imsave(filename, (right_blur * 255).astype(np.uint8))

def norm_for_display(img):
    img_min = img.min()
    img_max = img.max()
    return ((img - img_min) / (img_max - img_min + 1e-8))

def lighten(lap):
    lap_centered = lap - lap.mean()    
    max_abs = np.max(np.abs(lap_centered))  
    lap_scaled = lap_centered / (max_abs + 1e-8)

    lap_vis = (lap_scaled + 1) / 2.0

    gamma = 0.8 
    lap_vis = np.power(lap_vis, gamma)

    return (lap_vis * 255).astype(np.uint8)


def add_blue_tint(img, amount=0.05):
    img = img.copy().astype(np.float32)
    img[..., 2] += amount    # add 5% blue
    img = np.clip(img, 0, 1)
    return img

# print("start")
# total = []
# for orange, m in zip(left_lap, mask_stack):
#     result = blend_one(orange, m)
#     # result = norm_for_display(result)
#     vis = (norm_for_display(result) * 255).astype(np.uint8)
#     vis = add_blue_tint(vis, 0.15)
#     total.append(vis)
# save(total, "orange")

# total = []
# for apple, m in zip(right_lap, mask_stack):
#     result = blend_two(apple, m)
#     # result = norm_for_display(result)
#     vis = (norm_for_display(result) * 255).astype(np.uint8)
#     vis = add_blue_tint(vis, 0.15)
#     total.append(vis)
# save(total, "apple")

# total = []
# for apple, orange, m in zip(left_lap, right_lap, mask_stack):
#     result = blend(apple, orange, m)
#     result = lighten(result)
#     total.append(result)
# save(total, "final")

def get_pieced():

    total = []
    for apple, orange, m in zip(left_lap, right_lap, mask_stack):
        result = blend(apple, orange, m)
        total.append(result)

    final = combine(total)
    filename = f"2/test/final2.png"
    io.imsave(filename, (final * 255).astype(np.uint8))


    # i = 1
    # for im in mask_stack:
    #     im_norm = (im - im.min()) / (im.max() - im.min())
    #     filename = f"2/mask/mask_layer_{i}.png"
    #     io.imsave(filename, (im_norm * 255).astype(np.uint8))
    #     i += 1

# get_pieced()


def night_beach():
    bot = '2/my_blend/night.jpg'
    top = '2/my_blend/beach.jpg'
    top = io.imread(top)/255
    bot = io.imread(bot)/255

    top_stack = gauss_stack(top, 1)
    bot_stack = gauss_stack(bot, 1)

    top_lap = laplacian_stack(top_stack)
    bot_lap = laplacian_stack(bot_stack)

    mask = create_2_mask(top) 
    mask_stack = gauss_mask(mask, 1)

    total = []
    for a, b, m in zip(top_lap, bot_lap, mask_stack):
        result = blend(a, b, m)
        # result = lighten(result)
        total.append(result)

    filename = f"2/my_blend_results/night_beach2.png"
    io.imsave(filename, (result * 255).astype(np.uint8))

# night_beach()

def image_to_mask(img, invert=False):
    if img.ndim == 3:
        img = color.rgb2gray(img)
    if invert:
        mask = 1.0 - img
    else:
        mask = img
    return mask.astype(np.float32)

def camp():
    mask = '2/my_blend/mask.jpg'
    mask = io.imread(mask)/255
    mask = image_to_mask(mask)

    # plt.imshow(mask, cmap='gray')
    # plt.show()

    b = '2/my_blend/camp.jpg'
    a = '2/my_blend/dino.jpg'
    a = io.imread(a)/255
    b = io.imread(b)/255

    a_stack = gauss_stack(a, 0)
    b_stack = gauss_stack(b, 0)

    a_lap = laplacian_stack(a_stack)
    b_lap = laplacian_stack(b_stack)

    mask_stack = gauss_mask(mask, 0)

    total = []
    for a, b, m in zip(a_lap, b_lap, mask_stack):
        result = blend(a, b, m)
        total.append(result)

    filename = f"2/my_blend_results/historic.png"
    io.imsave(filename, (result * 255).astype(np.uint8))

# camp()

def ny():
    mask = '2/my_blend/m2.jpg'
    mask = io.imread(mask)/255
    mask = image_to_mask(mask)

    # plt.imshow(mask, cmap='gray')
    # plt.show()

    b = '2/my_blend/squirrel.jpg'
    a = '2/my_blend/ny.jpg'
    a = io.imread(a)/255
    b = io.imread(b)/255

    a_stack = gauss_stack(a, 0)
    b_stack = gauss_stack(b, 0)

    a_lap = laplacian_stack(a_stack)
    b_lap = laplacian_stack(b_stack)

    mask_stack = gauss_mask(mask, 0)

    total = []
    for a, b, m in zip(a_lap, b_lap, mask_stack):
        result = blend(a, b, m)
        total.append(result)

    filename = f"2/my_blend_results/ny_final.png"
    io.imsave(filename, (result * 255).astype(np.uint8))

# ny()