from detection import *
from transform import *


# # B1

# # get harris coords
# image = skio.imread("3/pair_pics/wheeler_2.jpg")
# h, coords1 = get_harris_corners(rgb2gray(image))  # output 2 x N 

# # print(coords1)

# # coords_1 = get_harris_coords(coords1) #change into (x, y) 
# # fig1 = display_points("3/pair_pics/wheeler_2.jpg", coords_1)
# # fig1.savefig("3/part_b/harris/harris_overlay.jpg", dpi=150)

# # ANMS
# anms_coords = anms_harris_corners(h, coords1) # takes in 2xN and outputs 2xN 
# # print(anms_coords) 
# # print("After ANMS:", anms_coords.shape)

# coords_plot = get_harris_coords(anms_coords) # change back to Nx2
# # print(coords_plot) 

# # fig2 = display_points("3/pair_pics/wheeler_2.jpg", coords_plot)
# # fig2.savefig("3/part_b/harris/harris_anms.jpg", dpi=150)

# # for y, x in coords_2:
# #     print(y, x)

# """
# ========================
# Features
# ========================
# """

# descriptors, patches, kept_coords = feature_descriptor("3/pair_pics/wheeler_2.jpg", coords_plot)

# fig, axes = plt.subplots(2, 5, figsize=(10, 4))
# for ax, p in zip(axes.flat, patches[:10]):
#     ax.imshow(p, cmap='gray')
#     ax.axis('off')
# plt.show()
# fig.savefig("3/part_b/feature/descriptors.jpg", dpi=150)


# # matching descriptors:


image1 = skio.imread("3/pair_pics/evan1.jpg")
image2 = skio.imread("3/pair_pics/evan2.jpg")

h1, coords1 = get_harris_corners(rgb2gray(image1))  # output 2 x N 
h2, coords2 = get_harris_corners(rgb2gray(image2))  # output 2 x N 

print("harris original corners: ", coords1.shape, coords2.shape)
anms_coords1 = anms_harris_corners(h1, coords1) # take in 2xN outputs 2xN
anms_coords2 = anms_harris_corners(h2, coords2)

anms_coords1 = get_harris_coords(anms_coords1)
anms_coords2 = get_harris_coords(anms_coords2)

descriptors1, patches1, kept_coords1 = feature_descriptor("3/pair_pics/evan1.jpg", anms_coords1) # takes in Nx2
descriptors2, patches2, kept_coords2 = feature_descriptor("3/pair_pics/evan2.jpg", anms_coords2) # takes in Nx2

# print(kept_coords1)

pic1_pts, pic2_pts = feature_matching(descriptors1, descriptors2)

a, b = feature_matching(descriptors2, descriptors1)

reverse_matches = {j2: j1 for j2, j1 in zip(a, b)}
mutual_img1 = []
mutual_img2 = []
for i1, i2 in zip(pic1_pts, pic2_pts):
    if reverse_matches.get(i2, -1) == i1:
        mutual_img1.append(kept_coords1[i1])  # Store coordinates, not indices
        mutual_img2.append(kept_coords2[i2])

print("Mutual matches in image1:", mutual_img1)
print("Mutual matches in image2:", mutual_img2)

# print(pic1_pts[:2])
# print("--------")
# print(pic2_pts[:2])

# matched_coords1 = get_points_from_idx(mutual_img1, kept_coords1)
# print("matched coords", matched_coords1)
# matched_coords2 = get_points_from_idx(mutual_img2, kept_coords2)
# print("matched coords", matched_coords2)

matched_coords1 = np.array(mutual_img1)
matched_coords2 = np.array(mutual_img2)

combined = np.hstack((image1, image2))
plt.figure(figsize=(10, 6))
plt.imshow(combined, cmap='gray')
for i, j in zip(matched_coords1, matched_coords2):
    y1, x1 = i[0], i[1]
    y2, x2 = j[0], j[1]
    x2 += image1.shape[1]
    plt.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
    plt.plot(x1, y1, 'bo', markersize=3)
    plt.plot(x2, y2, 'go', markersize=3)
plt.tight_layout()
# plt.savefig("3/part_b/feature/match_evan_ransac.jpg", dpi=150)
plt.show()


# #matches = matches[:10]

# for i, j in zip(mutual_img1, mutual_img2):
#     y1, x1 = kept_coords1[i]
#     y2, x2 = kept_coords2[j]
#     x2 += image1.shape[1]
#     plt.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
#     plt.plot(x1, y1, 'bo', markersize=3)
#     plt.plot(x2, y2, 'go', markersize=3)


# plt.title('Matched Features Between Images')
# plt.axis('off')
# plt.tight_layout()
# # plt.savefig("3/part_b/feature/match_evan.jpg", dpi=150)
# plt.show()


H, dest, warped, inl = ransac(matched_coords1, matched_coords2)

# print("ransac original h: ", H)

# compute_H = compute(warped, dest)
# print("compute: ", compute_H)

# print(inl)
# print("evan1 coords")
# print(matched_coords1[inl])
# print("evan2 coords")
# print(matched_coords2[inl])

# H2 = compute(matched_coords2[inl], matched_coords1[inl])
# print(H2)


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
# # plt.axis('off')
# plt.tight_layout()
# plt.savefig("3/part_b/feature/match_evan_ransac.jpg", dpi=150)
# plt.show()

# x = dest[:, 1]
# y = dest[:, 0]
# dest_swapped = np.column_stack([x, y])

# x = warped[:, 1]
# y = warped[:, 0]
# warped_swapped = np.column_stack([x, y])

# H = compute(warped_swapped, dest_swapped)

# display_points("3/pair_pics/evan1.jpg", dest_swapped)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# # Show image1 in ax1
# ax1.imshow(image1, cmap='gray')
# ax1.set_title("Image 1")

# # Show image2 in ax2
# ax2.imshow(image2, cmap='gray')
# ax2.set_title("Image 2")

# # Plot points on image1
# for i in dest:
#     y1, x1 = i[1], i[0]  # if coords are (x, y), careful with order
#     ax1.plot(x1, y1, 'bo', markersize=3)

# # Plot points on image2
# for j in warped:
#     y2, x2 = j[1], j[0]
#     ax2.plot(x2, y2, 'go', markersize=3)

# plt.tight_layout()
# plt.show()

width, height, x_off, y_off, H = find_dim("3/pair_pics/evan1.jpg", "3/pair_pics/evan2.jpg", H)
warped_image, mask = warpImageNearestNeighbor("3/pair_pics/evan2.jpg", H, width, height, x_off, y_off)
plot(warped_image)

image = blend_n("3/pair_pics/evan1.jpg", warped_image, mask, width, height, x_off, y_off, 30)
