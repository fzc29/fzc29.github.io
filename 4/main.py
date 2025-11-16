from calibration import * 
from nn_part1 import *
from neural_radiance import *
from viser_test import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

"""
=========================
Calibration
=========================
"""

test_folder = "4/arcu_cal/"
K, coeff = calibrate_cam(test_folder)

test_obj = "4/arcu_obj/"
poses, images = estimate_pose(test_obj, K, coeff)

save_as_dataset(images, poses, K, coeff)


# server = viser.ViserServer()

# for i in range(len(poses)):
#     H, W = images[i].shape[:2]
#     server.scene.add_camera_frustum(
#         f"/cameras/{i}", # give it a name
#         fov=2 * np.arctan2(H / 2, K[0, 0]), # field of view
#         aspect=W / H, # aspect ratio
#         scale=0.01, # scale of the camera frustum change if too small/big
#         wxyz=viser.transforms.SO3.from_matrix(poses[i][:3, :3]).wxyz, # orientation in quaternion format
#         position=poses[i][:3, 3], # position of the camera
#         image=images[i] # image to visualize
#     )
#     print("Frustum added for frame",i , flush=True)

# while True:
#     time.sleep(0.1)


"""
=============================
Fit NN to 2-D Image
=============================
"""
# hidden_dim = 256
# num_layers = 3
# batch_size = 10000
# lr = 1e-2
# iterations = 2000

# device = torch.device("mps")

# pixels, coords, H, W = load_image_and_coords("4/part1/fox.jpg")
# pixels, coords = pixels.to(device), coords.to(device)

# network = build_mlp(42, hidden_dim, num_layers).to(device)
# optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
# loss_fn = nn.MSELoss()

# for it in range(1, iterations + 1):
#     batch_coords, batch_colors = sample_batch(coords, pixels, batch_size)
#     enc_coords = positional_encoding(batch_coords)
    
#     pred_colors = network(enc_coords)
#     loss = loss_fn(pred_colors, batch_colors)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if it % 200 == 0 or it == 1:
#         with torch.no_grad():
#             full_pred = network(positional_encoding(coords))
#             mse_full = loss_fn(full_pred, pixels)
#             psnr_full = psnr(mse_full)
#         print(f"Iteration {it}/{iterations} - Loss: {loss.item():.6f}, PSNR: {psnr_full:.2f}dB")


# widths = [32, 64, 128, 256]
# L_values = [2, 5, 10, 15]  

# network = build_mlp(in_dim=2 + 4*L_pe, hidden_dim=width, num_layers=num_layers)


# L_Val = 10; Hidden Layers (2) = 256

# training("4/part1/fox.jpg", "fox", 2, 32, num_layers=3, batch_size=10000, lr=1e-2, iterations=5000) 
# training("4/part1/fox.jpg", "fox", 2, 256, num_layers=3, batch_size=10000, lr=1e-2, iterations=5000) 

# training("4/part1/fox.jpg", "fox", 10, 32, num_layers=3, batch_size=10000, lr=1e-2, iterations=5000) 
# training("4/part1/fox.jpg", "fox", 10, 256, num_layers=3, batch_size=10000, lr=1e-2, iterations=5000) 

# training("4/part1/lily.jpg", "lily", 2, 16, num_layers=3, batch_size=10000, lr=1e-2, iterations=5000) 
# training("4/part1/lily.jpg", "lily", 2, 256, num_layers=3, batch_size=10000, lr=1e-2, iterations=5000) 

# training("4/part1/lily.jpg", "lily", 10, 16, num_layers=3, batch_size=10000, lr=1e-2, iterations=5000) 
# training("4/part1/lily.jpg", "lily", 10, 256, num_layers=3, batch_size=10000, lr=1e-2, iterations=5000) 

# def render(sigmas, rgbs, step_size):
#     EPS = 1e-10
#     alpha = 1.0 - torch.exp(-sigmas * step_size)
#     ones = torch.ones((sigmas.shape[0], 1, 1), device=sigmas.device, dtype=sigmas.dtype)
#     cumprod_input = torch.cat([ones, 1.0 - alpha + EPS], dim=1)
#     cumprod = torch.cumprod(cumprod_input, dim=1)
#     T = cumprod[:, :-1, :]
#     weights = T * alpha
#     rendered = (weights * rgbs).sum(dim=1)
#     return rendered

# """
# =============================
# Fit NN from Multi-View Images
# =============================
# """

# print(f"\n{'='*60}")
# print("TRAINING NERF")
# print(f"{'='*60}\n")

# L_c = 10  # positional encoding frequency for xyz coords
# L_rd = 4   # positional encoding frequency for ray directions

# pos_in_dim = 3 + 2 * 3 * L_c   # = 3 + 2*3*10 = 63
# dir_in_dim = 3 + 2 * 3 * L_rd   # = 3 + 2*3*4  = 27
# hidden_dim = 256
# device = torch.device("mps")

# dataset_train = RaysData(images_train, K, c2ws_train)
# dataset_val = RaysData(images_val, K, c2ws_val)

# network =  NeRFNetwork(pos_in_dim, dir_in_dim, hidden_dim).to(device)
# optimizer = torch.optim.Adam(network.parameters(), lr=5e-4)
# loss_fn = nn.MSELoss()


# N_rays = 2048
# n_samples = 64
# near, far = 2.0, 6.0
# n_iter = 200


# step_size = (far - near) / n_samples
# history = {
#         'iterations': [],
#         'losses': [],
#         'psnrs': []
#     }

# os.makedirs('checkpoints', exist_ok=True)

# for iteration in range(n_iter):
#     # Sample rays
#     rays_o, rays_d, rgb_gt = sample_rays_global(images_train, c2ws_train, K, N_rays)
    
#     # Convert to torch
#     rays_o = torch.FloatTensor(rays_o).to(device)  # (B, 3)
#     rays_d = torch.FloatTensor(rays_d).to(device)  # (B, 3)
#     rgb_gt = torch.FloatTensor(rgb_gt).to(device)  # (B, 3)
    
#     # Sample points along rays
#     pts, z_vals = sample_points_along_rays(
#         rays_o.cpu().numpy(), rays_d.cpu().numpy(),
#         near=near, far=far, n_samples=n_samples, perturb=True
#     )
#     pts = torch.FloatTensor(pts).to(device)  # (B, N_samples, 3)
    
#     # Encode positions
#     B, N_samples, _ = pts.shape
#     pts_flat = pts.reshape(-1, 3)  # (B*N_samples, 3)
#     pts_encoded = positional_encoding(pts_flat, L_c)
   
#     # Encode directions (same direction for all samples along a ray)
#     dirs = rays_d.unsqueeze(1).expand(-1, N_samples, -1)  # (B, N_samples, 3)
#     dirs_flat = dirs.reshape(-1, 3)  # (B*N_samples, 3)
#     dirs_encoded = positional_encoding(dirs_flat, L_rd)
    
#     # Forward pass
#     rgb_pred, sigma_pred = network(pts_encoded, dirs_encoded)
    
#     # Reshape back
#     rgb_pred = rgb_pred.reshape(B, N_samples, 3)  # (B, N_samples, 3)
#     sigma_pred = sigma_pred.reshape(B, N_samples, 1)  # (B, N_samples, 1)
    
#     # Volume rendering
#     rgb_rendered = volrend(sigma_pred, rgb_pred, step_size)  # (B, 3)
    
#     # Compute loss
#     mse_loss = loss_fn(rgb_rendered, rgb_gt)
    
#     # Backward pass
#     optimizer.zero_grad()
#     mse_loss.backward()
#     optimizer.step()
    
#     # Compute PSNR
#     with torch.no_grad():
#         mse = mse_loss.item()
#         psnr = -10 * np.log10(mse)
    
#     # Log every 100 iterations
#     if (iteration + 1) % 20 == 0 or iteration == 0:
#         history['iterations'].append(iteration + 1)
#         history['losses'].append(mse_loss.item())
#         history['psnrs'].append(psnr)

#         print(f"Iter {iteration+1}/{n_iter} - Loss: {mse_loss.item():.6f} - PSNR: {psnr:.2f} dB")
        
#         if (iteration + 1) % 50 == 0 or iteration == 0:
#             print(f"  Saving checkpoint image...")
#             checkpoint_img = render_image(network, c2ws_val[0], K, H, W, device, L_c, L_rd, n_samples, near, far)
#             plt.imsave(f'checkpoints/iter_{iteration+1:04d}.png', checkpoint_img)

#     print(f"\n{'='*60}")
#     print("TRAINING COMPLETE!")
#     print(f"{'='*60}")
#     print(f"Final PSNR: {history['psnrs'][-1]:.2f} dB")
#     print(f"{'='*60}\n")
    
# print("\n" + "="*60)
# print("RENDERING NOVEL VIEW VIDEO")
# print("="*60 + "\n")

# frames = []
# for i, c2w in enumerate(c2ws_test):
#     print(f"Rendering frame {i+1}/{len(c2ws_test)}...")
#     img = render_image(network, c2w, K, H, W, device, L_c, L_rd, n_samples, near, far)
#     frames.append((img * 255).astype(np.uint8))

# # Also save as GIF (easier to view/share)
# imageio.mimsave('lego_view_video.gif', frames, fps=30, loop=0)
# print(f"✓ GIF saved to 'lego_view_video.gif'")

# # Save video
# imageio.mimsave('lego_view_video.mp4', frames, fps=30, codec='libx264')
# print(f"\n✓ Video saved to 'lego_view_video.mp4'")


# import cv2
# network.eval()
# with torch.no_grad():
#     images_rendered = []
    
#     for i, c2w in enumerate(c2ws_test):
#         H, W = images_train.shape[1:3]

#         # generate all pixel coordinates
#         u, v = np.meshgrid(np.arange(W), np.arange(H))
#         u = u.flatten() + 0.5
#         v = v.flatten() + 0.5
#         uv = np.stack([u, v], axis=-1)

#         # all rays for this camera
#         ray_o, ray_d = pixel_to_ray(K, c2w, uv)
        
#         # sample points along rays
#         points, _ = sample_points_along_rays(ray_o, ray_d, near, far, n_samples, perturb=False)

#         # flatten
#         points_flat = torch.from_numpy(points.reshape(-1, 3)).float().to(device)
#         dirs_flat = torch.from_numpy(np.repeat(ray_d[:, None, :], n_samples, axis=1).reshape(-1, 3)).float().to(device)

#         # positional encoding
#         pos_enc = positional_encoding(points_flat, L_c)
#         dir_enc = positional_encoding(dirs_flat, L_rd)

#         # predict RGB and sigma
#         rgbs_pred, sigmas_pred = network(pos_enc, dir_enc)
#         rgbs_pred = rgbs_pred.reshape(H*W, n_samples, 3)
#         sigmas_pred = sigmas_pred.reshape(H*W, n_samples, 1)

#         # render final color per ray
#         step_size = (far - near) / n_samples
#         rgb_rendered = render(sigmas_pred, rgbs_pred, step_size)

#         # reshape to image
#         final_img = rgb_rendered.cpu().numpy().reshape(H, W, 3)
#         images_rendered.append(final_img)

#         # save
#         img_save_path = f"video_lego/frame_{i:03d}.png"
#         plt.imsave(img_save_path, final_img)
#         print(f"Saved {img_save_path}")

# fps = 30  # frames per second
# video_path = os.path.join("video_lego", "lego_rendered.mp4")
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

# for frame in images_rendered:
#     frame_bgr = (frame * 255).astype(np.uint8)[..., ::-1]  # RGB → BGR for OpenCV
#     video_writer.write(frame_bgr)

# video_writer.release()
# print(f"Video saved → {video_path}")

# print(f"\n{'='*60}")
# print("ALL DONE!")
# print(f"{'='*60}")
# print(f"Check these files:")
# print(f"  - checkpoints/iter_*.png  (training progress images)")
# print(f"  - novel_view_video.mp4    (rotating view video)")
# print(f"  - novel_view_video.gif    (rotating view GIF)")
# print(f"{'='*60}\n")

# number of samples = 1024 rays
# 32 samples from each rays
# 200 iterations
# near far 2, 6


# 2000 iteration
# 1024 rays
# 32/64 samples per ray 

