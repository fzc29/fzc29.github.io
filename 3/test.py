from transform import *
from points import *

H = compute(evan2_arr, evan1_arr)
print("original H: ",H)
width, height, x_off, y_off, H = find_dim(evan1, evan2, H)
print("shifted H: ", H)
warped_image, mask = warpImageBilinear(evan2, H, width, height)
plot(warped_image)
# plot(mask)

image = blend_n(evan1, warped_image, mask, width, height, x_off, y_off, 30)
# save_image('3/new_pics/evan_blend.jpg', image)
plot(image)


# # Read images (float32 or uint8)
# imgA = read(wheeler_2)  # reference image
# imgB = read(wheeler_6)  # to be warped

# # Compute homography H (maps B → A coordinates)
# H = compute(wheeler_2_arr, wheeler_6_arr)

# # Find panorama canvas size
# final_w, final_h, x_off_B, y_off_B = find_panorama_canvas_size(imgA, imgB, H)

# # Warp imgB into panorama canvas
# warp_nn = cv2.warpPerspective(imgB, H, (final_w, final_h), flags=cv2.INTER_NEAREST)
# warp_bilinear = cv2.warpPerspective(imgB, H, (final_w, final_h), flags=cv2.INTER_LINEAR)

# # Place imgA in same canvas
# canvasA = np.zeros((final_h, final_w, 3), dtype=imgA.dtype)
# canvasA[-min_y : -min_y + imgA.shape[0], 
#         -min_x : -min_x + imgA.shape[1]] = imgA

"""
========================
Test
========================
"""
# H = compute(mc2_arr, mc4_arr)
# # def warpImageNearestNeighbor(im_path, H):
# warped_image, mask, x_off, y_off = warpImageBilinear(mc4, H)
# plot(warped_image)

# #def blend(reference, warped, mask, x_off, y_off, sigma):
# image = blend(mc2, warped_image, mask, x_off, y_off, 10)
# plot(image)

# H = compute(wheeler_6_arr, wheeler_2_arr)
# width, height, x_off, y_off, H = find_dim(wheeler_2, wheeler_6, H)
# warped_image, mask = warpImageNearestNeighbor(wheeler_6, H, width, height, x_off, y_off)
# plot(warped_image)
# plot(mask)

# blend_n(wheeler_2, warped_image, mask, width, height, x_off, y_off, 30)

H = compute(evan2_arr, evan1_arr)
print("original H: ",H)
width, height, x_off, y_off, H = find_dim(evan1, evan2, H)
print("shifted H: ", H)
warped_image, mask = warpImageNearestNeighbor(evan2, H, width, height, x_off, y_off)
plot(warped_image)
# plot(mask)

image = blend_n(evan1, warped_image, mask, width, height, x_off, y_off, 30)
# save_image('3/new_pics/evan_blend.jpg', image)
plot(image)



# Warp with nearest neighbor

# width, height, x_off, y_off, H = find_dim(wheeler_6, H)

# wheeler_6 = read(wheeler_6)
# warp_nn = cv2.warpPerspective(src=wheeler_6, M=H, dsize=(width, height),flags=cv2.INTER_NEAREST)

# plot(warp_nn)


# Warp with bilinear interpolation
# warp_bilinear = cv2.warpPerspective(src=im, M=H, dsize=(width, height),
#                                     flags=cv2.INTER_LINEAR)

#def blend(reference, warped, mask, x_off, y_off, sigma):
# image = blend(wheeler_2, warped_image, mask, x_off, y_off, 10)
# plot(image)













# test recification
# jimin = '3/rectification/jimin.jpg'
# points = np.array([(84,345),(144,414),(319,286),(233,244)])
# goal = np.array([[96,428],[216,428],[216,234],[96,234]])

# def rectify():
#     H = compute(points, goal)
#     display_points(jimin, points)

#     imwarped_nn, mask = warpImageBilinear(jimin, H)
#     # save_image("jimin_rectified", imwarped_nn)
#     plt.imshow(imwarped_nn)
#     plt.axis('off')
#     plt.show()

# rectify()
