import cv2
import numpy as np
import os 
import glob
import viser
import time 


def calibrate_cam(img_folder="", tag_size=0.02):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) 

    object_points = np.array([[0, 0, 0],    
                              [tag_size, 0, 0], 
                              [tag_size, tag_size, 0], 
                              [0, tag_size, 0]], dtype=np.float32)
    
    obj_3d_pts = [] # 3D points of object in world
    img_2d_pts = [] # 2D points of image
    img_size = None 

    img_files = glob.glob(os.path.join(img_folder, '*.[jJ][pP][gG]'))
    # print(f"Found {len(img_files)} calibration images")

    detections = 0 

    for img in img_files:
        img = cv2.imread(img)
        if img is None:
            print(f"Warning: Could not read {img}")
            continue

        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_size is None:
            h, w = img.shape[:2]
            img_size = (w, h)

        corners, ids, _ = detector.detectMarkers(img)
        if ids is not None and len(corners) > 0:
            detections += 1
            # print(f"detected {len(corners)} tags")

            for c in corners:
                img_2d = c[0].astype(np.float32)    # (4,2)
                img_2d_pts.append(img_2d)
                obj_3d_pts.append(object_points.copy())

            # cv2.aruco.drawDetectedCorners(img, corners, ids)
        else:
            print("No tags detected")

    print(f"\nSuccessfully detected tags in {detections}/{len(img_files)} images")

    print("\nCalibrating camera...")

    reprojection_err, camera_int_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(obj_3d_pts, img_2d_pts, img_size, None, None)

    print(f"\n{'='*60}")
    print(f"CALIBRATION COMPLETE!")
    print(f"{'='*60}")

    print(f"RMS Re-projection Error: {reprojection_err:.4f}")
    print(f"(Good calibration typically has error < 1.0)\n")

    print(f"Camera Matrix (K):")
    print(camera_int_matrix)
    print(f"\nFocal Length (fx, fy): ({camera_int_matrix[0,0]:.2f}, {camera_int_matrix[1,1]:.2f})")
    print(f"Principal Point (cx, cy): ({camera_int_matrix[0,2]:.2f}, {camera_int_matrix[1,2]:.2f})")

    print(f"\nDistortion Coefficients:")
    print(dist_coeffs)
    print(f"{'='*60}\n")
    
    return camera_int_matrix, dist_coeffs


def estimate_pose(obj_folder, camera_matrix, dist_coeffs, server=None, tag_size=0.056):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) 

    object_points = np.array([[0, 0, 0], 
                              [tag_size, 0, 0], 
                              [tag_size, tag_size, 0], 
                              [0, tag_size, 0]], dtype=np.float32)
    
    img_files = glob.glob(os.path.join(obj_folder, '*.[jJ][pP][gG]'))
    print(f"Found {len(img_files)} calibration images")

    successful = 0
    poses = []
    images = []

    for i, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img}")
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(img_gray)

        if ids is not None and len(corners) != 0:
            img_points = corners[0].reshape(-1, 2).astype(np.float32)
            print(f"detected {len(corners)} tags")
            success, rvec, tvec = cv2.solvePnP(
                    object_points,      # 3D points in world coordinates
                    img_points,       # 2D points in image coordinates
                    camera_matrix,      # Camera intrinsic matrix K
                    dist_coeffs,        # Distortion coefficients
                )

            if success:
                R, _ = cv2.Rodrigues(rvec)

                c2w = np.eye(4)
                c2w[:3, :3] = R.T
                c2w[:3, 3] = -R.T @ tvec.flatten() 

                poses.append(c2w)
                images.append(img)
                successful += 1

    print(f"\n{'='*60}")
    print(f"POSE ESTIMATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Successfully estimated {successful}/{len(img_files)} camera poses")
    print(f"{'='*60}\n")
    return poses, images 


def save_as_dataset(images_set, poses, camera_matrix, dist_coeffs):
    undistorted = []
    for img in images_set:
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs,  (w, h), 0, (w, h)) 

        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

        x, y, w_roi, h_roi = roi
        undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]

        new_camera_matrix[0, 2] -= x  # cx
        new_camera_matrix[1, 2] -= y  # cy

        undistorted.append(undistorted_img)

    # make the training sets 

    n_total = len(undistorted)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]

    images_train = np.array([undistorted[i] for i in train_indices], dtype=np.uint8)
    images_val = np.array([undistorted[i] for i in val_indices], dtype=np.uint8)

    c2ws_train = np.array([poses[i] for i in train_indices], dtype=np.float32)
    c2ws_val = np.array([poses[i] for i in val_indices], dtype=np.float32)
    c2ws_test = np.array([poses[i] for i in test_indices], dtype=np.float32)

    fx = new_camera_matrix[0, 0]
    fy = new_camera_matrix[1, 1]

    focal = float((fx + fy) / 2)

    np.savez(
        'myscaled_data.npz',
        images_train=images_train,    # (N_train, H, W, 3)
        c2ws_train=c2ws_train,        # (N_train, 4, 4)
        images_val=images_val,        # (N_val, H, W, 3)
        c2ws_val=c2ws_val,            # (N_val, 4, 4)
        c2ws_test=c2ws_test,          # (N_test, 4, 4)
        focal=focal                   # float
    )

    print(f"{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Training set:")
    print(f"  Images shape: {images_train.shape}")
    print(f"  Poses shape: {c2ws_train.shape}")
    print(f"  Image range: [{images_train.min()}, {images_train.max()}]")
    print(f"\nValidation set:")
    print(f"  Images shape: {images_val.shape}")
    print(f"  Poses shape: {c2ws_val.shape}")
    print(f"\nTest set:")
    print(f"  Poses shape: {c2ws_test.shape}")
    print(f"  (Test images not saved - used for novel view synthesis)")
    print(f"\nFocal length: {focal:.2f}")
    print(f"{'='*60}\n")






