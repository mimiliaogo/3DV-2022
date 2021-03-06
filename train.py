import time
import torch
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
import src.losses as losses
from src.losses import ChamferDistanceLoss

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
# mesh
from pytorch3d.ops import sample_points_from_meshes

from pytorch3d.structures import Meshes

# for mesh return 
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2

cd_loss = ChamferDistanceLoss()

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

@hydra.main(config_path="configs/", config_name="config.yml")
def train_model(cfg: DictConfig):
    print(cfg.data_dir)
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
    train_loader = iter(loader)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()
    # plot loss
    running_loss = 0.0
    avg_loss = []
    loss_freq = 20

    if cfg.load_checkpoint:
        checkpoint = torch.load(f'{cfg.base_dir}/checkpoint_{cfg.dtype}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting training !")
    for step in range(start_iter, cfg.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        
        if cfg.dtype == 'mesh':
            mesh_dict = next(train_loader)
            ground_truth_3d = Meshes(
                verts=mesh_dict["verts"],
                faces=mesh_dict["faces_idx"],
                # textures=mesh_dict["textures"]
            )
            images_gt = mesh_dict["images"]
        else:
            images_gt, ground_truth_3d, _ = next(train_loader)

        images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, cfg)
        loss = calculate_loss(prediction_3d, ground_truth_3d, cfg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        running_loss += loss_vis

        if (step % cfg.save_freq) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{cfg.base_dir}/checkpoint_{cfg.dtype}.pth')
        if step != 0 and (step % loss_freq) == 0:
            avg_loss.append(running_loss / loss_freq)
            running_loss = 0.0

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.5f" % (step, cfg.max_iter, total_time, read_time, iter_time, loss_vis))

    # plot the loss curve
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.plot(avg_loss)
    plt.savefig(f'{cfg.base_dir}/loss_curve_{cfg.dtype}.png')
    print('Done!')


if __name__ == '__main__':
    train_model()