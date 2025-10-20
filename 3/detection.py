from transform import *
from harris import * 
from points import *
from skimage.color import rgb2gray
from skimage import transform


# trial to read 
def get_harris_coords(coords):
    return coords[::-1].T

# anms
def anms_harris_corners(h, coords, c_robust=0.9, nip=2000):
    strengths = h[coords[0], coords[1]]
    N = coords.shape[1]

    radii = np.full(N, np.inf)

    for i in range(N):
        stronger_points = strengths > c_robust * strengths[i]
        stronger_coords = coords[:, stronger_points].T
        current_coord = coords[:, i].reshape(1, 2) 

        # print("current cord", current_coord.shape)
        # print("stronger: ", stronger_coords.shape)

        if np.any(stronger_points):
            dists_sq = dist2(current_coord, stronger_coords)
            radii[i] = np.sqrt(np.min(dists_sq))

    index = np.argsort(-radii,)[:nip]
    return coords[:, index]


def feature_descriptor(image, coords):
    # use grayscale image 
    img = read(image, gray=True)
  
    descriptors = []
    patches = []
    kept_coords = []

    for y, x in coords:
        y, x = int(y), int(x)
        if y - 20 < 0 or y + 20 >= img.shape[0] or x - 20 < 0 or x + 20 >= img.shape[1]:
            continue 
        
        # sample 40 x 40 windows around each point 
        patch = img[y-20: y+20, x-20: x+20]
        # down sample into 8x8 
        patch_small = transform.resize(patch, (8, 8), anti_aliasing=True)

        # normalize
        mu = np.mean(patch_small)
        sigma = np.std(patch_small)
        if sigma < 1e-6:  # division by 0 errors appeared 
            continue
        patch_norm = (patch_small - mu) / sigma

        descriptors.append(patch_norm.flatten())
        patches.append(patch_norm)
        kept_coords.append((y, x))

    return np.array(descriptors), patches, kept_coords


def feature_matching(descriptor_1, descriptor_2, threshold=0.8):
    curr = []
    match = []

    for i in range(descriptor_1.shape[0]):
        distances = np.linalg.norm(descriptor_2 - descriptor_1[i], axis=1)  # positive distances with length d2
        # print("distances:", distances)
        idx = np.argsort(distances)
        best_dist = distances[idx[0]]
        # print(idx[0])
        # print(best_dist)
        second_best_dist = distances[idx[1]]
        # print(second_best_dist)

        # Apply Lowe's ratio test
        if best_dist < threshold * second_best_dist:
            curr.append(i)
            match.append(idx[0])

    # find the distance between each descriptor 
    return curr, match

def get_points_from_idx(idx, coords):
    coords_matched = []
    for i in idx:
        coords_matched.append(coords[i])
    return np.array(coords_matched)

def swap(array):
    x = array[:, 1]
    y = array[:, 0]
    return np.column_stack([x, y])

# coords are lists of tuples 
def ransac(dest_coords, warped_coords, iterations=10000, threshold=15):
    n = dest_coords.shape[0]

    best_inliers = []
    best_H = None

    dest_coords = swap(dest_coords)
    warped_coords = swap(warped_coords)

    for i in range(iterations):
        # randomly pick 4 points
        random_coords_idx = np.random.choice(n, 4, replace=False)

        dest_pts = get_points_from_idx(random_coords_idx, dest_coords)
        warped_pts = get_points_from_idx(random_coords_idx, warped_coords)

        # compute H 
        H = compute(warped_pts, dest_pts)
        #print(H)

        # apply homography (project points)
        pts1_hom = np.hstack([warped_coords, np.ones((n, 1))])

        proj_hom = (H @ pts1_hom.T)
        proj_cartesian = (proj_hom[:2, :] / proj_hom[2, :]).T
        error = np.linalg.norm(proj_cartesian - dest_coords, axis=1)

        # count inliers 
        inl = np.where(error < threshold)[0]

        # check if max inlier amount, if update else skip 
        if len(inl) > len(best_inliers):
            best_inliers = inl
            best_H = H

    # recompute H 
    dest_final = np.array(get_points_from_idx(best_inliers, dest_coords))
    warped_final = np.array(get_points_from_idx(best_inliers, warped_coords))
    best_H = compute(warped_final, dest_final)
    return best_H, dest_final, warped_final, best_inliers


