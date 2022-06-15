import time
import torch
import torch.nn as nn
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
import src.losses as losses
from src.losses import ChamferDistanceLoss
import numpy as np

import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt
# for voxel visualize
from mpl_toolkits.mplot3d import Axes3D

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2

import cv2

cd_loss = ChamferDistanceLoss()
# for voxel prediction
sidmoid = nn.Sigmoid()

def calculate_loss(predictions, ground_truth, cfg):
    if cfg.dtype == 'voxel':
        loss = losses.voxel_loss(predictions,ground_truth)
    elif cfg.dtype == 'point':
        loss = cd_loss(predictions, ground_truth)
    elif cfg.dtype == 'mesh':
            sample_trg = sample_points_from_meshes(ground_truth, cfg.n_points)
            sample_pred = sample_points_from_meshes(predictions, cfg.n_points)

            loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
            loss_smooth = losses.smoothness_loss(predictions, cfg)

            loss = cfg.w_chamfer * loss_reg + loss_smooth        
    return loss

# This is the visualizatio code for mesh/voxel/point-cloud
def visualize(rend, step, cfg, save_type=''):
    fig = plt.figure()
    if cfg.dtype == 'point':
        ax = fig.add_subplot(projection='3d')
        ax.scatter(rend[...,0], rend[...,1], rend[...,2], c='r', marker='.')
    if cfg.dtype == 'voxel':
        rend = rend.squeeze().__ge__(0.5)
        ax = fig.gca(projection=Axes3D.name)
        ax.set_aspect('auto')
        ax.voxels(rend, edgecolor="k")
    elif cfg.dtype == 'mesh':
        points = sample_points_from_meshes(rend, 5000)
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        ax.scatter3D(x, z, -y)
        ax.view_init(190, 30)

    
    plt.savefig(f'{cfg.base_dir}/vis-report/{step}_{cfg.dtype}_{save_type}.png')


@hydra.main(config_path="configs/", config_name="config.yml")
def evaluate_model(cfg: DictConfig):
    shapenetdb = ShapeNetDB(cfg.data_dir, cfg.dtype)


    if cfg.dtype == "mesh":
        loader = torch.utils.data.DataLoader(
            shapenetdb,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate_batched_R2N2,
            pin_memory=True,
            drop_last=True)
    else:
        loader = torch.utils.data.DataLoader(
            shapenetdb,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True)

    eval_loader = iter(loader)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.eval()

    start_iter = 0
    start_time = time.time()

    avg_loss = []

    if cfg.load_eval_checkpoint:
        checkpoint = torch.load(f'{cfg.base_dir}/checkpoint_{cfg.dtype}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        if cfg.dtype == 'mesh':
            mesh_dict = next(eval_loader)
            ground_truth_3d = Meshes(
                verts=mesh_dict["verts"],
                faces=mesh_dict["faces_idx"],
                # textures=mesh_dict["textures"]
            )
            images_gt = mesh_dict["images"]
        else:
            images_gt, ground_truth_3d, _ = next(eval_loader)

        images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()

        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, cfg)
        
        if cfg.dtype == 'point':
            torch.save(prediction_3d.detach().cpu(), f'{cfg.base_dir}/pre_point_cloud.pt')

        loss = calculate_loss(prediction_3d, ground_truth_3d, cfg).cpu().item()
        

        # TODO: visualization 
        if (step % cfg.vis_freq) == 0:
            # visualization block
            if cfg.dtype == "mesh":
                rend = prediction_3d.cpu().detach()[0]
                rend2 = ground_truth_3d.cpu().detach()[0]
            else:
                rend = prediction_3d.cpu().detach().numpy()[0]
                rend2 = ground_truth_3d.cpu().detach().numpy()[0]
            
            visualize(rend, step, cfg)
            visualize(rend2, step, cfg, 'gt')
            # save raw image
            cv2.imwrite(f'{cfg.base_dir}/vis-report/image_{step}.png', images_gt.cpu().detach().numpy()[0]*255)

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        avg_loss.append(loss)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); eva_loss: %.3f" % (step, cfg.max_iter, total_time, read_time, iter_time, torch.tensor(avg_loss).mean()))

    print('Done!')

if __name__ == '__main__':
    evaluate_model()