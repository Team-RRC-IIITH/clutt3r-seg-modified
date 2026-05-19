"""Instance-wise scene update and rigid-body motion optimization."""


from pathlib import Path


import argparse
import pickle
import glob

import numpy as np
import cv2
import open3d as o3d
import torch
import copy
from scipy import ndimage
import torch.nn.functional as F
from scipy.spatial import cKDTree


from .utils import (
    load_update_frame_data, refine_masks_by_depth, r6d_to_mat, load_update_gt_data
)


def optimize_rigid_body_only_z_motion(points, colors, update_frames, inst_id, args,
                               device="cuda", num_iterations=100, lr=0.01,
                               use_original_pcd=False, original_data=None,
                               matching_info=None, instance_mask_path=None,
                               output_dir=None, verbose=True, visualize=True):

    def Rz_from_theta(theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.stack([
            torch.stack([ c, -s, torch.tensor(0., device=theta.device)]),
            torch.stack([ s,  c, torch.tensor(0., device=theta.device)]),
            torch.stack([torch.tensor(0., device=theta.device), torch.tensor(0., device=theta.device), torch.tensor(1., device=theta.device)])
        ], dim=0)
        return R

    def Rx_from_theta(theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.stack([
            torch.stack([torch.tensor(1., device=theta.device), torch.tensor(0., device=theta.device), torch.tensor(0., device=theta.device)]),
            torch.stack([torch.tensor(0., device=theta.device),  c, -s]),
            torch.stack([torch.tensor(0., device=theta.device),  s,  c])
        ], dim=0)
        return R


    def rasterize_and_calculate_iou(posed_points_np, frame_data, target_mask_np):
        H, W = frame_data['img_shape']
        K = frame_data['K']
        pose_w2c = frame_data['pose']
        R_cam, t_cam = pose_w2c[:3, :3], pose_w2c[:3, 3]


        points_cam = (R_cam @ posed_points_np.T + t_cam.reshape(3, 1)).T


        valid_z = points_cam[:, 2] > 0
        if not np.any(valid_z):
            return 0.0

        points_cam = points_cam[valid_z]


        u = (points_cam[:, 0] * K[0, 0] / points_cam[:, 2]) + K[0, 2]
        v = (points_cam[:, 1] * K[1, 1] / points_cam[:, 2]) + K[1, 2]


        valid_proj = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(valid_proj):
            return 0.0

        u, v = u[valid_proj].astype(int), v[valid_proj].astype(int)


        pred_mask = np.zeros((H, W), dtype=bool)
        pred_mask[v, u] = True


        intersection = np.logical_and(pred_mask, target_mask_np).sum()
        union = np.logical_or(pred_mask, target_mask_np).sum()

        return intersection / (union + 1e-8)


    if use_original_pcd and original_data is not None:

        inst2all_points = original_data["inst2all_points"]

        if inst2all_points is not None and inst_id in inst2all_points:

            all_points_data = inst2all_points[inst_id]
            points_np = all_points_data[:, :3]
            colors_np = all_points_data[:, 3:]
        else:
            raise ValueError(f"inst2all_points not provided for instance {inst_id}")

    else:

        points_np = points.cpu().numpy()
        colors_np = colors.cpu().numpy()
    points_tensor = torch.from_numpy(points_np).float().to(device)
    colors_tensor = torch.from_numpy(colors_np).float().to(device)


    initial_centroid = np.mean(points_np, axis=0)
    initial_centroid_tensor = torch.from_numpy(initial_centroid).float().to(device)

    theta_z = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
    tvec = torch.tensor(initial_centroid, device=device, requires_grad=True, dtype=torch.float32)


    optimizer = torch.optim.Adam([
        {'params': [theta_z], 'lr': lr * 3.0},
        {'params': [tvec], 'lr': lr * 1.0}
    ])


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    stage1_iterations = int(num_iterations * 0.3)
    early_stop_iou_threshold = 0.7

    for i in range(num_iterations):

        if i < stage1_iterations:

            R = torch.eye(3, device=device, dtype=torch.float32)


            lambda_chamfer = 50.0
            lambda_photometric = 0.5
            lambda_txy = 0.0
            lambda_tz = 20.0

        else:

            R = Rz_from_theta(theta_z)


            lambda_chamfer = 20.0
            lambda_photometric = 1.5
            lambda_txy = 1.0
            lambda_tz = 1.0

        optimizer.zero_grad()


        points_centered = points_tensor - initial_centroid_tensor
        posed_points = (R @ points_centered.T).T + tvec

        total_loss = 0
        num_valid_frames = 0

        for frame_data in update_frames:

            lid = frame_data['matched_lid']


            if matching_info is not None:

                fidx = frame_data['fidx']


                if inst_id not in matching_info or (fidx, lid) not in matching_info[inst_id]:
                    raise ValueError(f"Data consistency error: Mask for instance {inst_id} (lid={lid}) in frame {fidx} is not listed in matching_info.")


            mask_glob = glob.glob(str((instance_mask_path) / f"mask_{frame_data['fidx']:06d}_*.png"))
            mask_refined = refine_masks_by_depth(mask_glob, None, mode="naive")

            if lid not in mask_refined:
                raise ValueError(f"Mask not found for instance {inst_id} (lid={lid}) in frame {frame_data['fidx']}")


            target_mask = mask_refined[lid]
            H, W = frame_data['img_shape']


            K = torch.from_numpy(frame_data['K']).float().to(device)
            pose_w2c = torch.from_numpy(frame_data['pose']).float().to(device)
            R_cam, t_cam = pose_w2c[:3, :3], pose_w2c[:3, 3]
            target_mask_tensor = torch.from_numpy(target_mask).float().to(device)


            points_cam = (R_cam @ posed_points.T + t_cam.unsqueeze(1)).T


            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]


            u = (points_cam[:, 0] * fx / points_cam[:, 2] + cx)
            v = (points_cam[:, 1] * fy / points_cam[:, 2] + cy)


            valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_cam[:, 2] > 0)

            if not torch.any(valid_mask):
                continue


            proj_u = u[valid_mask]
            proj_v = v[valid_mask]
            pts_colors = colors_tensor[valid_mask]

            projected_points_2d = torch.stack([proj_u, proj_v], dim=1)
            contour_np = ndimage.binary_erosion(target_mask) ^ target_mask
            contour_coords = torch.from_numpy(np.stack(np.where(contour_np)[::-1], axis=1).astype(np.float32)).to(device)

            dist_p2c, _ = torch.cdist(projected_points_2d, contour_coords).min(dim=1)
            dist_c2p, _ = torch.cdist(contour_coords, projected_points_2d).min(dim=1)
            chamfer_loss = dist_p2c.mean() + dist_c2p.mean()


            H, W = frame_data['img_shape']
            u_norm = (proj_u / (W - 1) * 2.0) - 1.0
            v_norm = (proj_v / (H - 1) * 2.0) - 1.0


            grid = torch.stack([u_norm, v_norm], dim=1).view(1, -1, 1, 2)


            img = frame_data['image_tensor'].to(device).unsqueeze(0)
            mask = target_mask_tensor.unsqueeze(0).unsqueeze(0)


            sampled_rgb = F.grid_sample(img, grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=True)
            sampled_rgb = sampled_rgb.view(3, -1).transpose(0, 1)

            sampled_mw  = F.grid_sample(mask, grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=True)
            sampled_mw  = sampled_mw.view(-1)


            eps = 1e-6
            per_point_l1 = (sampled_rgb - pts_colors).abs().sum(dim=1)
            w = sampled_mw.clamp(min=0.0, max=1.0)
            photometric_loss = (w * per_point_l1).sum() / (w.sum() + eps)


            t_vec_offset = tvec - initial_centroid_tensor

            reg_loss = (lambda_txy * (t_vec_offset[:2].norm()) +
                        lambda_tz  * (t_vec_offset[2].abs()))

            frame_loss = lambda_chamfer * (chamfer_loss / max(H, W)) + lambda_photometric * photometric_loss + reg_loss
            total_loss += frame_loss
            num_valid_frames += 1

            if visualize and i % 10 == 0:
                target_img_np = (frame_data['image_tensor'].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                rgb_visualize_img = cv2.cvtColor(target_img_np, cv2.COLOR_RGB2BGR)


                proj_pts_np = projected_points_2d.detach().cpu().numpy().astype(int)
                proj_colors_np = colors_tensor[valid_mask].detach().cpu().numpy()
                for (px, py), color in zip(proj_pts_np, proj_colors_np):
                    cv2.circle(rgb_visualize_img, (px, py), 2, (color * 255).tolist()[::-1], -1)


                rgb_visualize_img[contour_np] = [255, 0, 0]


                visualize_dir = output_dir / "visualize_masks"
                visualize_dir.mkdir(parents=True, exist_ok=True)
                visualize_path = visualize_dir / f"visualize_inst{inst_id}_frame{frame_data['fidx']}_iter{i}.png"
                cv2.imwrite(str(visualize_path), rgb_visualize_img)

        if num_valid_frames > 0:
            total_loss = total_loss / num_valid_frames
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([theta_z, tvec], max_norm=1.0)
            optimizer.step()
            scheduler.step()

        if i == stage1_iterations - 1 and verbose:
            instance_iou_scores = []


            current_posed_points_np = posed_points.detach().cpu().numpy()


            for frame_data in update_frames:

                lid = frame_data['matched_lid']
                mask_glob = glob.glob(str((instance_mask_path) / f"mask_{frame_data['fidx']:06d}_*.png"))
                mask_refined = refine_masks_by_depth(mask_glob, None, mode="naive")
                if lid not in mask_refined:
                    continue
                target_mask_np = mask_refined[lid].astype(bool)


                iou = rasterize_and_calculate_iou(current_posed_points_np, frame_data, target_mask_np)
                instance_iou_scores.append(iou)


            if instance_iou_scores:
                avg_iou_for_instance = np.mean(instance_iou_scores)
                if avg_iou_for_instance > early_stop_iou_threshold:
                    break


    optimized_R = Rz_from_theta(theta_z).detach()

    optimized_t = tvec.detach()

    return optimized_R, optimized_t


def optimize_rigid_body_motion(points, colors, update_frames, inst_id, args,
                               device="cuda", num_iterations=100, lr=0.01,
                               use_original_pcd=False, original_data=None,
                               matching_info=None, instance_mask_path=None,
                               output_dir=None, verbose=True, visualize=True):
    def rasterize_and_calculate_iou(posed_points_np, frame_data, target_mask_np):
        H, W = frame_data['img_shape']
        K = frame_data['K']
        pose_w2c = frame_data['pose']
        R_cam, t_cam = pose_w2c[:3, :3], pose_w2c[:3, 3]


        points_cam = (R_cam @ posed_points_np.T + t_cam.reshape(3, 1)).T


        valid_z = points_cam[:, 2] > 0
        if not np.any(valid_z):
            return 0.0

        points_cam = points_cam[valid_z]


        u = (points_cam[:, 0] * K[0, 0] / points_cam[:, 2]) + K[0, 2]
        v = (points_cam[:, 1] * K[1, 1] / points_cam[:, 2]) + K[1, 2]


        valid_proj = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(valid_proj):
            return 0.0

        u, v = u[valid_proj].astype(int), v[valid_proj].astype(int)


        pred_mask = np.zeros((H, W), dtype=bool)
        pred_mask[v, u] = True


        intersection = np.logical_and(pred_mask, target_mask_np).sum()
        union = np.logical_or(pred_mask, target_mask_np).sum()

        return intersection / (union + 1e-8)


    if use_original_pcd and original_data is not None:

        inst2all_points = original_data["inst2all_points"]

        if inst2all_points is not None and inst_id in inst2all_points:

            all_points_data = inst2all_points[inst_id]
            points_np = all_points_data[:, :3]
            colors_np = all_points_data[:, 3:]
        else:
            raise ValueError(f"inst2all_points not provided for instance {inst_id}")

    else:

        points_np = points.cpu().numpy()
        colors_np = colors.cpu().numpy()
    points_tensor = torch.from_numpy(points_np).float().to(device)
    colors_tensor = torch.from_numpy(colors_np).float().to(device)


    initial_centroid = np.mean(points_np, axis=0)
    initial_centroid_tensor = torch.from_numpy(initial_centroid).float().to(device)

    r6d = torch.tensor([1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    r6d = r6d + 1e-4 * torch.randn_like(r6d)
    r6d.requires_grad_()
    tvec = torch.tensor(initial_centroid, device=device, requires_grad=True, dtype=torch.float32)


    optimizer = torch.optim.Adam([
        {'params': [r6d], 'lr': lr * 3.0},
        {'params': [tvec], 'lr': lr * 1.0}
    ])


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    stage1_iterations = int(num_iterations * 0.3)
    early_stop_iou_threshold = 0.7

    for i in range(num_iterations):

        if i < stage1_iterations:

            R = torch.eye(3, device=device, dtype=torch.float32)


            lambda_chamfer = 50.0
            lambda_photometric = 0.5
            lambda_txy = 0.0
            lambda_tz = 20.0

        else:

            R = r6d_to_mat(r6d)


            lambda_chamfer = 20.0
            lambda_photometric = 1.5
            lambda_txy = 1.0
            lambda_tz = 1.0

        optimizer.zero_grad()


        points_centered = points_tensor - initial_centroid_tensor
        posed_points = (R @ points_centered.T).T + tvec

        total_loss = 0
        num_valid_frames = 0

        for frame_data in update_frames:

            lid = frame_data['matched_lid']


            if matching_info is not None:

                fidx = frame_data['fidx']


                if inst_id not in matching_info or (fidx, lid) not in matching_info[inst_id]:
                    raise ValueError(f"Data consistency error: Mask for instance {inst_id} (lid={lid}) in frame {fidx} is not listed in matching_info.")


            mask_glob = glob.glob(str((instance_mask_path) / f"mask_{frame_data['fidx']:06d}_*.png"))
            mask_refined = refine_masks_by_depth(mask_glob, None, mode="naive")

            if lid not in mask_refined:
                raise ValueError(f"Mask not found for instance {inst_id} (lid={lid}) in frame {frame_data['fidx']}")


            target_mask = mask_refined[lid]
            H, W = frame_data['img_shape']


            K = torch.from_numpy(frame_data['K']).float().to(device)
            pose_w2c = torch.from_numpy(frame_data['pose']).float().to(device)
            R_cam, t_cam = pose_w2c[:3, :3], pose_w2c[:3, 3]
            target_mask_tensor = torch.from_numpy(target_mask).float().to(device)


            points_cam = (R_cam @ posed_points.T + t_cam.unsqueeze(1)).T


            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]


            u = (points_cam[:, 0] * fx / points_cam[:, 2] + cx)
            v = (points_cam[:, 1] * fy / points_cam[:, 2] + cy)


            valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_cam[:, 2] > 0)

            if not torch.any(valid_mask):
                continue


            proj_u = u[valid_mask]
            proj_v = v[valid_mask]
            pts_colors = colors_tensor[valid_mask]

            projected_points_2d = torch.stack([proj_u, proj_v], dim=1)
            contour_np = ndimage.binary_erosion(target_mask) ^ target_mask
            contour_coords = torch.from_numpy(np.stack(np.where(contour_np)[::-1], axis=1).astype(np.float32)).to(device)

            dist_p2c, _ = torch.cdist(projected_points_2d, contour_coords).min(dim=1)
            dist_c2p, _ = torch.cdist(contour_coords, projected_points_2d).min(dim=1)
            chamfer_loss = dist_p2c.mean() + dist_c2p.mean()


            H, W = frame_data['img_shape']
            u_norm = (proj_u / (W - 1) * 2.0) - 1.0
            v_norm = (proj_v / (H - 1) * 2.0) - 1.0


            grid = torch.stack([u_norm, v_norm], dim=1).view(1, -1, 1, 2)


            img = frame_data['image_tensor'].to(device).unsqueeze(0)
            mask = target_mask_tensor.unsqueeze(0).unsqueeze(0)


            sampled_rgb = F.grid_sample(img, grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=True)
            sampled_rgb = sampled_rgb.view(3, -1).transpose(0, 1)

            sampled_mw  = F.grid_sample(mask, grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=True)
            sampled_mw  = sampled_mw.view(-1)


            eps = 1e-6
            per_point_l1 = (sampled_rgb - pts_colors).abs().sum(dim=1)
            w = sampled_mw.clamp(min=0.0, max=1.0)
            photometric_loss = (w * per_point_l1).sum() / (w.sum() + eps)


            t_vec_offset = tvec - initial_centroid_tensor

            reg_loss = (lambda_txy * (t_vec_offset[:2].norm()) +
                        lambda_tz  * (t_vec_offset[2].abs()))

            frame_loss = lambda_chamfer * (chamfer_loss / max(H, W)) + lambda_photometric * photometric_loss + reg_loss
            total_loss += frame_loss
            num_valid_frames += 1

            if visualize and i % 10 == 0:
                target_img_np = (frame_data['image_tensor'].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                rgb_visualize_img = cv2.cvtColor(target_img_np, cv2.COLOR_RGB2BGR)


                proj_pts_np = projected_points_2d.detach().cpu().numpy().astype(int)
                proj_colors_np = colors_tensor[valid_mask].detach().cpu().numpy()
                for (px, py), color in zip(proj_pts_np, proj_colors_np):
                    cv2.circle(rgb_visualize_img, (px, py), 2, (color * 255).tolist()[::-1], -1)


                rgb_visualize_img[contour_np] = [255, 0, 0]


                visualize_dir = output_dir / "visualize_masks"
                visualize_dir.mkdir(parents=True, exist_ok=True)
                visualize_path = visualize_dir / f"visualize_inst{inst_id}_frame{frame_data['fidx']}_iter{i}.png"
                cv2.imwrite(str(visualize_path), rgb_visualize_img)

        if num_valid_frames > 0:
            total_loss = total_loss / num_valid_frames
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([r6d, tvec], max_norm=1.0)
            optimizer.step()
            scheduler.step()

        if i == stage1_iterations - 1 and verbose:
            instance_iou_scores = []


            current_posed_points_np = posed_points.detach().cpu().numpy()


            for frame_data in update_frames:

                lid = frame_data['matched_lid']
                mask_glob = glob.glob(str((instance_mask_path) / f"mask_{frame_data['fidx']:06d}_*.png"))
                mask_refined = refine_masks_by_depth(mask_glob, None, mode="naive")
                if lid not in mask_refined:
                    continue
                target_mask_np = mask_refined[lid].astype(bool)


                iou = rasterize_and_calculate_iou(current_posed_points_np, frame_data, target_mask_np)
                instance_iou_scores.append(iou)


            if instance_iou_scores:
                avg_iou_for_instance = np.mean(instance_iou_scores)
                if avg_iou_for_instance > early_stop_iou_threshold:
                    break


    optimized_R = r6d_to_mat(r6d).detach()
    optimized_t = tvec.detach()

    return optimized_R, optimized_t

def instance_wise_scene_update(args, initial_data, matching_info, use_original_pcd=True, original_data=None, target_id=-1, update=-1, visualize=False, verbose=False):
    database_root = args.experiment_data_dir
    output_dir = database_root / "output"
    scene_data_path = database_root / "data"
    instance_mask_path = scene_data_path / "instance_masks"
    if not matching_info:
        raise ValueError(
            "Scene update received empty matching_info. "
            "Run update segmentation on a frame with Grounded-SAM masks before scene update."
        )

    initial_instance_pcds = initial_data["instance_pcds"]
    initial_instance_original_pcds = original_data["inst2all_points"]

    initial_instances = set(initial_instance_pcds.keys())


    matched_instances = set(matching_info.keys())


    seg_update_instances = set(initial_data["node2inst"].values())


    consistent_instances = matched_instances
    new_instances = seg_update_instances - initial_instances
    removed_instances = initial_instances - matched_instances

    def rasterize_points_to_mask(pts_world, K, R_cam, t_cam, img_shape):
        H, W = img_shape
        pts_cam = (R_cam @ pts_world.T + t_cam.reshape(3, 1)).T
        z = pts_cam[:, 2]
        valid = z > 0
        if not np.any(valid):
            return np.zeros((H, W), dtype=bool)

        u = (pts_cam[valid, 0] * K[0, 0] / z[valid]) + K[0, 2]
        v = (pts_cam[valid, 1] * K[1, 1] / z[valid]) + K[1, 2]

        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        mask = np.zeros((H, W), dtype=bool)
        mask[v[in_img], u[in_img]] = True
        return mask

    iou_thresh = getattr(args, "iou_thresh", 0.75)

    update_needed_instances = set()
    skipped_instances        = set()

    for inst_id in sorted(consistent_instances):
        if inst_id not in initial_instance_original_pcds:
            continue
        pcd_np = initial_instance_original_pcds[inst_id][:, :3]
        iou_list = []


        for fidx, lid in matching_info[inst_id]:
            frame_data = load_update_frame_data(scene_data_path, fidx, verbose)
            if frame_data is None:
                continue


            mask_glob = glob.glob(
                str(instance_mask_path / f"mask_{fidx:06d}_*.png"))
            mask_refined = refine_masks_by_depth(mask_glob, None, mode="naive")
            if lid not in mask_refined:
                continue
            tgt_mask = mask_refined[lid].astype(bool)


            K        = frame_data["K"]
            pose_w2c = frame_data["pose"]
            R_cam    = pose_w2c[:3, :3]
            t_cam    = pose_w2c[:3, 3]
            pred_mask = rasterize_points_to_mask(
                pcd_np, K, R_cam, t_cam, frame_data["img_shape"])


            inter = np.logical_and(pred_mask, tgt_mask).sum()
            union = np.logical_or (pred_mask, tgt_mask).sum()
            iou   = inter / (union + 1e-8)
            iou_list.append(iou)


            if visualize:
                visualize_dir = Path(output_dir) / "iou_visualize_masks"
                visualize_dir.mkdir(parents=True, exist_ok=True)

                H, W = pred_mask.shape
                visualize_img = np.zeros((H, W, 3), dtype=np.uint8)
                pred_color = np.array([0, 0, 255], dtype=np.uint8)
                tgt_color = np.array([0, 255, 0], dtype=np.uint8)
                intersection_color = np.array([255, 0, 0], dtype=np.uint8)
                intersection = np.logical_and(pred_mask, tgt_mask)

                visualize_img[pred_mask] = pred_color
                visualize_img[tgt_mask] = tgt_color
                visualize_img[intersection] = intersection_color

                visualize_filename = visualize_dir / f"inst_{inst_id:03d}_frame_{fidx:06d}_lid_{lid:03d}.png"
                cv2.imwrite(str(visualize_filename), visualize_img)


        mean_iou = np.mean(iou_list) if iou_list else 1.0
        if mean_iou < iou_thresh:
            update_needed_instances.add(inst_id)
        else:
            skipped_instances.add(inst_id)

    optimized_poses = {}
    target_filtered_update_needed_instances = []
    z_only = True
    if target_id not in consistent_instances:
        target_filtered_update_needed_instances = update_needed_instances
    elif target_id >= 0:

        if target_id in update_needed_instances:
            target_filtered_update_needed_instances = [target_id]
    else:

        target_filtered_update_needed_instances = update_needed_instances

    if target_filtered_update_needed_instances:
        device = "cuda" if torch.cuda.is_available() else "cpu"


        for inst_id in target_filtered_update_needed_instances:
            pcd_np = initial_instance_pcds[inst_id]
            points = torch.from_numpy(pcd_np[:, :3]).float().to(device)
            colors = torch.from_numpy(pcd_np[:, 3:]).float().to(device)


            matching_data = matching_info[inst_id]
            update_frames = []

            for fidx, lid in matching_data:
                update_frame_data = load_update_frame_data(scene_data_path, fidx, verbose)
                if update_frame_data:

                    update_frame_data['matched_lid'] = lid
                    update_frames.append(update_frame_data)

            if update_frames:

                if z_only:
                    optimized_R, optimized_t = optimize_rigid_body_only_z_motion(
                        points, colors, update_frames, inst_id,
                        args, device=device, use_original_pcd=use_original_pcd,
                        original_data=original_data, matching_info=matching_info,
                        instance_mask_path = instance_mask_path, output_dir=output_dir,
                        verbose=verbose, visualize=visualize
                    )

                    optimized_poses[inst_id] = {
                        "R": optimized_R.cpu().numpy(),
                        "t": optimized_t.cpu().numpy(),
                        "status": "optimized",
                        "matching_frames": [fidx for fidx, _ in matching_data]
                    }

                else:
                    optimized_R, optimized_t = optimize_rigid_body_motion(
                        points, colors, update_frames, inst_id,
                        args, device=device, use_original_pcd=use_original_pcd,
                        original_data=original_data, matching_info=matching_info,
                        instance_mask_path = instance_mask_path, output_dir=output_dir,
                        verbose=verbose, visualize=visualize
                    )

                    optimized_poses[inst_id] = {
                        "R": optimized_R.cpu().numpy(),
                        "t": optimized_t.cpu().numpy(),
                        "status": "optimized",
                        "matching_frames": [fidx for fidx, _ in matching_data]
                    }
    successfully_updated_instances = set()


    for inst_id, pose_info in optimized_poses.items():

        pcd_orig_np = initial_instance_original_pcds[inst_id][:, :3]
        initial_centroid = np.mean(pcd_orig_np, axis=0)


        R_opt = pose_info["R"]
        t_opt = pose_info["t"]

        xyz_centered = pcd_orig_np - initial_centroid
        xyz_rotated = (R_opt @ xyz_centered.T).T
        xyz_new = xyz_rotated + t_opt


        iou_list_after_opt = []
        for fidx, lid in matching_info[inst_id]:
            frame_data = load_update_frame_data(scene_data_path, fidx, verbose)
            if frame_data is None:
                continue


            mask_glob = glob.glob(str(instance_mask_path / f"mask_{fidx:06d}_*.png"))
            mask_refined = refine_masks_by_depth(mask_glob, None, mode="naive")
            if lid not in mask_refined:
                continue
            tgt_mask = mask_refined[lid].astype(bool)


            K, pose_w2c = frame_data["K"], frame_data["pose"]
            R_cam, t_cam = pose_w2c[:3, :3], pose_w2c[:3, 3]
            pred_mask_after_opt = rasterize_points_to_mask(
                xyz_new, K, R_cam, t_cam, frame_data["img_shape"]
            )


            inter = np.logical_and(pred_mask_after_opt, tgt_mask).sum()
            union = np.logical_or(pred_mask_after_opt, tgt_mask).sum()
            iou = inter / (union + 1e-8)
            iou_list_after_opt.append(iou)

        if iou_list_after_opt:
            mean_iou_after_opt = np.mean(iou_list_after_opt)
            if mean_iou_after_opt >= iou_thresh:
                successfully_updated_instances.add(inst_id)

    final_instance_pcds = copy.deepcopy(initial_instance_original_pcds)


    for inst_id in removed_instances:
        if inst_id in final_instance_pcds:
            del final_instance_pcds[inst_id]


    for inst_id in consistent_instances:
        if inst_id not in final_instance_pcds:
            continue

        pcd_orig = final_instance_pcds[inst_id][:, :3]
        initial_centroid = np.mean(pcd_orig, axis=0)

        if inst_id in optimized_poses:

            pose = optimized_poses[inst_id]
            R, t_new = pose["R"], pose["t"]
            xyz_centered = pcd_orig - initial_centroid
            xyz_rotated = (R @ xyz_centered.T).T
            xyz_new = xyz_rotated + t_new
            final_instance_pcds[inst_id][:, :3] = xyz_new


    xyz_scene_final_list, rgb_scene_final_list = [], []
    for pcd_data in final_instance_pcds.values():
        if pcd_data.size > 0:
            xyz_scene_final_list.append(pcd_data[:, :3])
            rgb_scene_final_list.append(pcd_data[:, 3:])

    xyz_scene_final = None
    if xyz_scene_final_list:
        xyz_scene_final = np.vstack(xyz_scene_final_list)

    depth_evaluation_results = {}
    if target_id != -1:
        def calculate_chamfer_distance(pcd_a, pcd_b):
            if pcd_a is None or pcd_b is None or pcd_a.shape[0] == 0 or pcd_b.shape[0] == 0: return None
            b_tree = cKDTree(pcd_b)
            dist_a_to_b, _ = b_tree.query(pcd_a)
            a_tree = cKDTree(pcd_a)
            dist_b_to_a, _ = a_tree.query(pcd_b)
            return np.mean(dist_a_to_b) + np.mean(dist_b_to_a)

        def project_pcd_to_depth(pcd, K, R_cam, t_cam, img_shape):
            H, W = img_shape
            if pcd is None or pcd.shape[0] == 0: return np.full((H, W), np.inf, dtype=np.float32)

            pts_cam = (R_cam @ pcd.T + t_cam.reshape(3, 1)).T
            valid_z_mask = pts_cam[:, 2] > 1e-3
            if not np.any(valid_z_mask): return np.full((H, W), np.inf, dtype=np.float32)

            pts_cam_valid = pts_cam[valid_z_mask]
            u = (pts_cam_valid[:, 0] * K[0, 0] / pts_cam_valid[:, 2]) + K[0, 2]
            v = (pts_cam_valid[:, 1] * K[1, 1] / pts_cam_valid[:, 2]) + K[1, 2]
            z_pred = pts_cam_valid[:, 2]
            u_int, v_int = np.round(u).astype(int), np.round(v).astype(int)
            valid_proj_mask = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
            if not np.any(valid_proj_mask): return np.full((H, W), np.inf, dtype=np.float32)

            u_valid, v_valid = u_int[valid_proj_mask], v_int[valid_proj_mask]
            z_pred_valid = z_pred[valid_proj_mask]

            depth_map = np.full((H, W), np.inf, dtype=np.float32)
            sorted_indices = np.argsort(-z_pred_valid)
            depth_map[v_valid[sorted_indices], u_valid[sorted_indices]] = z_pred_valid[sorted_indices]
            return depth_map


        xyz_scene_initial_list = [pcd[:, :3] for pcd in initial_instance_original_pcds.values() if pcd.size > 0]
        xyz_scene_initial = np.vstack(xyz_scene_initial_list) if xyz_scene_initial_list else None
        pcd_before_opt = initial_instance_original_pcds.get(target_id, np.array([]))[:, :3]


        pcd_after_opt = final_instance_pcds.get(target_id, np.array([]))[:, :3]


        all_scene_errors_2d = []
        all_target_errors_2d_before, all_target_errors_2d_after = [], []
        all_chamfer_errors_3d_before, all_chamfer_errors_3d_after = [], []


        for fidx, lid in matching_info.get(target_id, []):
            frame_data = load_update_gt_data(scene_data_path, fidx, verbose=False)
            if not frame_data or 'depth' not in frame_data or frame_data['depth'] is None: continue

            gt_depth = frame_data['depth']
            H, W = gt_depth.shape

            mask_glob = glob.glob(str(instance_mask_path / f"mask_{fidx:06d}_*.png"))
            mask_refined = refine_masks_by_depth(mask_glob, None, mode="naive")
            if lid not in mask_refined: continue
            target_gt_mask = mask_refined[lid].astype(bool)

            K, pose_w_to_c = frame_data["K"], frame_data["pose"]
            R_cam, t_cam = pose_w_to_c[:3, :3], pose_w_to_c[:3, 3]


            pred_depth_map_before = project_pcd_to_depth(xyz_scene_initial, K, R_cam, t_cam, (H, W))
            pred_depth_map_after = project_pcd_to_depth(xyz_scene_final, K, R_cam, t_cam, (H, W))


            eval_mask_scene = (gt_depth > 0) & (pred_depth_map_after != np.inf)
            if np.any(eval_mask_scene):
                all_scene_errors_2d.extend(np.abs(gt_depth[eval_mask_scene] - pred_depth_map_after[eval_mask_scene]))

            eval_mask_target_before = (target_gt_mask) & (gt_depth > 0) & (pred_depth_map_before != np.inf)
            if np.any(eval_mask_target_before):
                all_target_errors_2d_before.extend(np.abs(gt_depth[eval_mask_target_before] - pred_depth_map_before[eval_mask_target_before]))

            eval_mask_target_after = (target_gt_mask) & (gt_depth > 0) & (pred_depth_map_after != np.inf)
            if np.any(eval_mask_target_after):
                all_target_errors_2d_after.extend(np.abs(gt_depth[eval_mask_target_after] - pred_depth_map_after[eval_mask_target_after]))


            valid_gt_mask = (target_gt_mask) & (gt_depth > 0)
            if not np.any(valid_gt_mask): continue
            v_gt, u_gt = np.where(valid_gt_mask)
            z_gt = gt_depth[v_gt, u_gt]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            pcd_cam_gt = np.vstack(((u_gt - cx) * z_gt / fx, (v_gt - cy) * z_gt / fy, z_gt)).T
            pcd_gt_target = (R_cam.T @ (pcd_cam_gt.T - t_cam.reshape(3, 1))).T

            dist_before = calculate_chamfer_distance(pcd_before_opt, pcd_gt_target)
            if dist_before is not None: all_chamfer_errors_3d_before.append(dist_before)

            dist_after = calculate_chamfer_distance(pcd_after_opt, pcd_gt_target)
            if dist_after is not None: all_chamfer_errors_3d_after.append(dist_after)


        if all_scene_errors_2d:
            depth_evaluation_results['scene_mae_meters'] = np.mean(all_scene_errors_2d)
        if all_target_errors_2d_before:
            depth_evaluation_results['target_mae_meters_before_opt'] = np.mean(all_target_errors_2d_before)
        if all_target_errors_2d_after:
            depth_evaluation_results['target_mae_meters_after_opt'] = np.mean(all_target_errors_2d_after)
        if all_chamfer_errors_3d_before:
            depth_evaluation_results['target_chamfer_dist_meters_before_opt'] = np.mean(all_chamfer_errors_3d_before)
        if all_chamfer_errors_3d_after:
            depth_evaluation_results['target_chamfer_dist_meters_after_opt'] = np.mean(all_chamfer_errors_3d_after)


    updated_instance_pcds = final_instance_pcds

    if xyz_scene_final is not None:
        rgb_scene_final = np.vstack(rgb_scene_final_list)

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(xyz_scene_final.astype(np.float64))
        scene_pcd.colors = o3d.utility.Vector3dVector(rgb_scene_final.astype(np.float64))

        export_path = Path(output_dir) / f"updated_scene_{update}.ply"
        o3d.io.write_point_cloud(str(export_path), scene_pcd)
        update_grasp_input_all_pcd = (xyz_scene_final, rgb_scene_final)
    else:
        raise RuntimeError("Updated scene point cloud is empty; no PLY was written.")


    export_instance_pcds = {}
    for inst_id, pcd_data in updated_instance_pcds.items():
        if pcd_data.size > 0:
            export_instance_pcds[inst_id] = (pcd_data[:, :3], pcd_data[:, 3:])
        else:
            export_instance_pcds[inst_id] = (np.array([]), np.array([]))


    scene_update_data = {
        "instance_pcds": export_instance_pcds,
        "optimized_poses": optimized_poses,
        "matching_info": matching_info,
        "not_moved_instances": skipped_instances,
        "moved_instances": update_needed_instances,
        "target_filtered_update_needed_instances": target_filtered_update_needed_instances,
        "successfully_updated_instances": successfully_updated_instances,
        "update_grasp_input_all_pcd": update_grasp_input_all_pcd,
        "depth_evaluation_results": depth_evaluation_results,
    }

    return scene_update_data


def main(args):
    state_path = args.experiment_data_dir / "state.pkl"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing {state_path}. Run initial segmentation before scene update.")

    with open(state_path, 'rb') as f:
        state = pickle.load(f)

    instance_wise_scene_update(
        args,
        initial_data=state,
        matching_info=state['updates'][f'update_{args.update_num}']['matching_info'],
        use_original_pcd=True,
        original_data=state.get("original_data"),
        update=args.update_num,
        visualize=args.visualize,
        verbose=False
    )


if __name__ == "__main__":
    pa = argparse.ArgumentParser()


    pa.add_argument("--experiment_data_dir", type=Path, required=True, help="Directory where state.pkl is located and will be updated.")
    pa.add_argument("--update_num", type=int, required=True, help="Which update sequence this scene update corresponds to.")
    pa.add_argument("--update_fid", type=int, required=True, help="Update frame index.")
    pa.add_argument("--visualize", action='store_true',
                   help="If set, save visualization outputs")

    args = pa.parse_args()

    main(args)
