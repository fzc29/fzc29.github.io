from transform import *
from points import *
from detection import *

"""
========================
Part 1
========================
"""

def homograph(im1, im2, arr1, arr2):
    H = compute(arr2, arr1)
    width, height, x_off, y_off, H = find_dim(im1, im2, H)
    warped_image, mask = warpImageBilinear(im2, H, width, height)
    image = blend_n(im1, warped_image, mask, width, height, x_off, y_off, 30)
    return image, H
   
# im, H = homograph(bancroft2, bancroft5, bancroft2_arr, bancroft5_arr)
# im = im[:, im.shape[1]//2:]
# save_image('3/finalized/part1/hom_bancroft.jpg', im)

# im, H = homograph(evan1, evan2, evan1_arr, evan2_arr)
# save_image('3/finalized/part1/hom_evan.jpg', im)

# im, H = homograph(grime1, grime2, grime1_arr, grime2_arr)
# save_image('3/finalized/part1/hom_grime.jpg', im)

# im, H = homograph(mc2, mc4, mc2_arr, mc4_arr)
# save_image('3/finalized/part1/hom_mc.jpg', im)

# im, H = homograph(wheeler_2, wheeler_6, wheeler_2_arr, wheeler_6_arr)
# save_image('3/finalized/part1/hom_wheeler.jpg', im)

im1 = '3/pair_pics/stan1.jpg'
im2 = '3/pair_pics/stan2.jpg'
arr1 = np.array([[450,136],[601,128],[665,125],[326,336],[470,181],[572,175]])
arr2 = np.array([[264,139],[401,142],[449,145],[120,350],[284,183],[376,181]])
im, H = homograph(im1, im2, arr1, arr2)
save_image('3/finalized/part1/hom_stan.jpg', im)

# d = display_points(evan1, evan1_arr)
# b = display_points(evan2, evan2_arr)
# print(H)
# d.savefig('3/finalized/part1/display_pts/evan1.jpg', dpi=150)
# b.savefig('3/finalized/part1/display_pts/evan2.jpg', dpi=150)

# d = display_points(wheeler_2, wheeler_2_arr)
# b = display_points(wheeler_6, wheeler_6_arr)
# print(H)
# d.savefig('3/finalized/part1/display_pts/wh2.jpg', dpi=150)
# b.savefig('3/finalized/part1/display_pts/wh6.jpg', dpi=150)


def rectify(goal, points, pic):
    H = compute(points, goal)
    fig = display_points(pic, points)
    im = read(pic)
    height, width = im.shape[:2]
    image, _ = warpImageBilinear(pic, H, width, height)
    plot(image)
    return fig, image

# fig, im = rectify(card_goal, card_pts, card)
# fig.savefig("3/finalized/part1/card_points.jpg", dpi=150)
# save_image('3/finalized/part1/rec_card.jpg', im)

# fig, im = rectify(jimin_goal, jimin_pts, jimin)
# fig.savefig("3/finalized/part1/jimin_points.jpg", dpi=150)
# save_image('3/finalized/part1/rec_jimin.jpg', im)

import time
def timer(im1, im2, arr1, arr2):
    H = compute(arr2, arr1)
    width, height, x_off, y_off, H = find_dim(im1, im2, H)

    start_time = time.time() 
    im1, _ =warpImageNearestNeighbor(im2, H, width, height)  
    end_time = time.time() 
    nn = end_time - start_time

    start_time = time.time() 
    im2, _ = warpImageBilinear(im2, H, width, height)  
    end_time = time.time() 
    bil = end_time - start_time

    return nn, bil, im1, im2

# nn, bil, im1, im2 = timer(evan1, evan2, evan1_arr, evan2_arr)
# print("evan nn: ", nn)
# print("evan bil: ", bil)
# save_image('3/finalized/part1/warp/e_nn.jpg', im1)
# save_image('3/finalized/part1/warp/e_bil.jpg', im2)

# nn, bil, im1, im2 = timer(grime1, grime2, grime1_arr, grime2_arr)
# print("grime nn: ", nn)
# print("grime bil: ", bil)
# save_image('3/finalized/part1/warp/g_nn.jpg', im1)
# save_image('3/finalized/part1/warp/g_bil.jpg', im2)

# nn, bil, im1, im2 = timer(mc2, mc4, mc2_arr, mc4_arr)
# print("mc nn: ", nn)
# print("mc bil: ", bil)
# save_image('3/finalized/part1/warp/m_nn.jpg', im1)
# save_image('3/finalized/part1/warp/m_bil.jpg', im2)



"""
========================
Part 2
========================
"""

def harris(img):
    image = skio.imread(img)
    h, coords1 = get_harris_corners(rgb2gray(image))
    coords_1 = get_harris_coords(coords1)
    fig1 = display_points(img, coords_1)

    anms_coords = anms_harris_corners(h, coords1)
    coords_plot = get_harris_coords(anms_coords)
    fig2 = display_points(img, coords_plot)

    return fig1, fig2

# fig1, fig2 = harris(wheeler_2)
# fig1.savefig("3/finalized/part2/harris/wh_harris.jpg", dpi=150)
# fig2.savefig("3/finalized/part2/harris/wh_anms.jpg", dpi=150)

# im1 = '3/pair_pics/stan1.jpg'
# im2 = '3/pair_pics/stan2.jpg'

# image1 = skio.imread(im1)
# image2 = skio.imread(im2)

# h1, coords1 = get_harris_corners(rgb2gray(image1))  # output 2 x N 
# h2, coords2 = get_harris_corners(rgb2gray(image2))  # output 2 x N 


# anms_coords1 = anms_harris_corners(h1, coords1) # take in 2xN outputs 2xN
# anms_coords2 = anms_harris_corners(h2, coords2)
# anms_coords1 = get_harris_coords(anms_coords1)
# anms_coords2 = get_harris_coords(anms_coords2)


# descriptors1, patches1, kept_coords1 = feature_descriptor(im1, anms_coords1) # takes in Nx2
# descriptors2, patches2, kept_coords2 = feature_descriptor(im2, anms_coords2) # takes in Nx2

# # fig, axes = plt.subplots(2, 5, figsize=(10, 4))

# # for i, ax in enumerate(axes.flat):
# #     ax.imshow(patches1[i], cmap='gray')
# #     ax.set_title(f"{i}")
# #     ax.axis('off')

# # plt.suptitle("First 10 Feature Descriptor Patches for EVANS", fontsize=14)
# # plt.tight_layout()
# # plt.show()

# # fig.savefig("3/finalized/part2/feature/evan_descriptor.jpg", dpi=150)

# pic1_pts, pic2_pts = feature_matching(descriptors1, descriptors2)

# a, b = feature_matching(descriptors2, descriptors1)

# reverse_matches = {j2: j1 for j2, j1 in zip(a, b)}
# mutual_img1 = []
# mutual_img2 = []
# for i1, i2 in zip(pic1_pts, pic2_pts):
#     if reverse_matches.get(i2, -1) == i1:
#         mutual_img1.append(kept_coords1[i1])  # Store coordinates, not indices
#         mutual_img2.append(kept_coords2[i2])

# matched_coords1 = np.array(mutual_img1)
# matched_coords2 = np.array(mutual_img2)

# combined = np.hstack((image1, image2))
# fig = plt.figure(figsize=(10, 6))
# plt.imshow(combined, cmap='gray')
# for i, j in zip(matched_coords1, matched_coords2):
#     y1, x1 = i[0], i[1]
#     y2, x2 = j[0], j[1]
#     x2 += image1.shape[1]
#     plt.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
#     plt.plot(x1, y1, 'bo', markersize=3)
#     plt.plot(x2, y2, 'go', markersize=3)
# plt.tight_layout()
# # plt.title("Matched Features of EVANS", fontsize=14)
# plt.show()
# # # fig.savefig("3/finalized/part2/feature/matched_evan.jpg", dpi=150)

# H, dest, warped, inl = ransac(matched_coords1, matched_coords2)

# combined = np.hstack((image1, image2))
# plt.figure(figsize=(10, 6))
# plt.imshow(combined, cmap='gray')
# for i, j in zip(dest, warped):
#     y1, x1 = i[0], i[1]
#     y2, x2 = j[0], j[1]
#     x2 += image1.shape[1]
#     plt.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
#     plt.plot(x1, y1, 'bo', markersize=3)
#     plt.plot(x2, y2, 'go', markersize=3)
# plt.tight_layout()
# plt.show()

# width, height, x_off, y_off, H = find_dim(im1, im2, H)
# warped_image, mask = warpImageNearestNeighbor(im2, H, width, height)
# image = blend_n(im1, warped_image, mask, width, height, x_off, y_off, 30)

# save_image('3/finalized/part2/ransac/hom_stan.jpg', image)


# may need cv_env environment)