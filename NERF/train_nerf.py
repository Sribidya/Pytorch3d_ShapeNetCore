import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm
import time
import csv
import os

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)
from utils.plot_image_grid import image_grid
from utils.generate_cow_renders import generate_cow_renders
from utils.helper_functions import (generate_rotating_nerf,
                                    huber,
                                    show_full_render,
                                    sample_images_at_mc_locs)
from nerf_model import NeuralRadianceField

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180) 
print(f'Generated {len(target_images)} images/silhouettes/cameras.')

render_size = target_images.shape[1] * 2

volume_extent_world = 3.0

raysampler_mc = MonteCarloRaysampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

raymarcher = EmissionAbsorptionRaymarcher()
renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)

render_size = target_images.shape[1] * 2
volume_extent_world = 3.0
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size,
    image_width=render_size,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world)

raymarcher = EmissionAbsorptionRaymarcher()
renderer_grid = ImplicitRenderer(
    raysampler=raysampler_grid, raymarcher=raymarcher)

neural_radiance_field = NeuralRadianceField()

torch.manual_seed(1)
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)
target_cameras = target_cameras.to(device)
target_images = target_images.to(device)
target_silhouettes = target_silhouettes.to(device)
neural_radiance_field = neural_radiance_field.to(device)

lr = 1e-3
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr) 
batch_size = 6
n_iter = 3000


loss_history_color, loss_history_sil = [], []
for iteration in range(n_iter):
    if iteration == round(n_iter * 0.75):
        print('Decreasing LR 10-fold ...')
        optimizer = torch.optim.Adam(
            neural_radiance_field.parameters(), lr=lr * 0.1
        )

    optimizer.zero_grad()
    batch_idx = torch.randperm(len(target_cameras))[:batch_size]

    batch_cameras = FoVPerspectiveCameras(
        R = target_cameras.R[batch_idx],
        T = target_cameras.T[batch_idx],
        znear = target_cameras.znear[batch_idx],
        zfar = target_cameras.zfar[batch_idx],
        aspect_ratio = target_cameras.aspect_ratio[batch_idx],
        fov = target_cameras.fov[batch_idx],
        device = device)

    rendered_images_silhouettes, sampled_rays = renderer_mc(
        cameras=batch_cameras,
        volumetric_function=neural_radiance_field
    )

    rendered_images, rendered_silhouettes = (
        rendered_images_silhouettes.split([3, 1], dim=-1)
    )

    silhouettes_at_rays = sample_images_at_mc_locs(
        target_silhouettes[batch_idx, ..., None],
        sampled_rays.xys
    )

    sil_err = huber(
        rendered_silhouettes,
        silhouettes_at_rays,
    ).abs().mean()
    colors_at_rays = sample_images_at_mc_locs(
        target_images[batch_idx],
        sampled_rays.xys
    )

    color_err = huber(
        rendered_images,
        colors_at_rays,
    ).abs().mean()

    loss = color_err + sil_err
    loss_history_color.append(float(color_err))
    loss_history_sil.append(float(sil_err))

    loss.backward()
    optimizer.step()

    # Visualize the full renders every 100 iterations.
    if iteration % 100 == 0:
        show_idx = torch.randperm(len(target_cameras))[:1]
        fig = show_full_render(
            neural_radiance_field,
            FoVPerspectiveCameras(
                R = target_cameras.R[show_idx], 
                T = target_cameras.T[show_idx], 
                znear = target_cameras.znear[show_idx],
                zfar = target_cameras.zfar[show_idx],
                aspect_ratio = target_cameras.aspect_ratio[show_idx],
                fov = target_cameras.fov[show_idx],
                device = device,
            ), 
            target_images[show_idx][0],
            target_silhouettes[show_idx][0],
            renderer_grid,
            loss_history_color,
            loss_history_sil,
        )
        fig.savefig(f'intermediate_{iteration}')
    if iteration % 500 == 0:
        print(f"[Iter {iteration}] Color loss: {color_err.item():.4f} | Silhouette loss: {sil_err.item():.4f}")

with torch.no_grad():
    rotating_nerf_frames = generate_rotating_nerf(
        neural_radiance_field,
        target_cameras,
        renderer_grid,
        n_frames=3*5,
        device=device
    ) 

#image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), rows=3, cols=5, rgb=True, fill=True)
frames = (rotating_nerf_frames.clamp(0., 1.) * 255).byte().cpu().numpy()
imageio.mimsave("nerf_rotation.gif", frames, fps=10)
# Save as MP4 using ffmpeg writer
writer = imageio.get_writer("nerf_rotation.mp4", fps=10, codec="libx264", quality=8)
for f in frames:
    writer.append_data(f)
writer.close()
print("Saved rotation video: nerf_rotation.gif / nerf_rotation.mp4")


def evaluate_model(model, cameras, images, renderer, device, label="Evaluation"):
    """
    Evaluate NeRF on a set of cameras/images.
    Computes MSE, PSNR, and SSIM with progress bar, timing, and CSV logging.
    """
    psnr_vals, ssim_vals, mse_vals = [], [], []

    print(f"\nRunning {label} ...")
    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(len(cameras)), desc=f"{label} Progress", ncols=100):
            # Safe camera indexing
            rendered, _ = renderer(
                cameras=cameras[[i]],
                #volumetric_function=lambda rays: model.batched_forward(rays, n_batches=8)
                volumetric_function=lambda ray_bundle, **kwargs: model.batched_forward(ray_bundle, n_batches=8)


            )

            pred = rendered[..., :3][0].clamp(0, 1).cpu()
            gt = images[i].cpu()

            # --- NEW: Resize ground truth if shapes don't match ---
            if pred.shape != gt.shape:
                 gt = torch.nn.functional.interpolate(
                    gt.permute(2, 0, 1).unsqueeze(0),  # (H, W, C) -> (1, C, H, W)
                    size=pred.shape[:2],
                    mode='bilinear',
                    align_corners=False
                  ).squeeze(0).permute(1, 2, 0)  # back to (H, W, C)

            # --- MSE
            mse = torch.mean((pred - gt) ** 2).item()
            mse_vals.append(mse)

            # --- PSNR
            psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse)))
            psnr_vals.append(psnr.item())

            # --- SSIM
            ssim_val = ssim_metric(gt.numpy(), pred.numpy(), channel_axis=-1, data_range=1.0)
            ssim_vals.append(ssim_val)

    total_time = time.time() - start_time

    mean_mse = np.mean(mse_vals)
    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)

    print(f"\n=== {label} Results ===")
    print(f"  MSE  = {mean_mse:.6f}")
    print(f"  PSNR = {mean_psnr:.2f} dB")
    print(f"  SSIM = {mean_ssim:.4f}")
    print(f"  Time = {total_time/60:.2f} minutes")

    # === CSV logging ===
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "nerf_evaluation_log.csv")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Label", "MSE", "PSNR (dB)", "SSIM", "Time (min)", "Timestamp"])
        writer.writerow([label, mean_mse, mean_psnr, mean_ssim, total_time / 60, time.strftime("%Y-%m-%d %H:%M:%S")])

    print(f" Results saved to {csv_path}")

# --- Run evaluation ---
evaluate_model(neural_radiance_field, target_cameras, target_images, renderer_grid, device, label="Evaluation")
