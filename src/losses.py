import torch
import torch.nn as nn
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
# define losses
def voxel_loss(voxel_src,voxel_tgt):
    # voxel_src: [B, 33, 33, 33], voxel_tgt: [B, 33, 33, 33]
	# implement some loss for binary voxel grids
    # sidmoid = nn.Sigmoid() # => in model
    bce_loss = nn.BCELoss()
    prob_loss = bce_loss(voxel_src, voxel_tgt.float())
    return prob_loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# implement chamfer loss from scratch
    cd_loss = ChamferDistanceLoss()
    loss_chamfer = cd_loss(point_cloud_src, point_cloud_tgt)

    return loss_chamfer

# enforce smoothness by adding shape regularizers
def smoothness_loss(mesh_src, cfg):
	# implement laplacian smoothening loss
    
    # the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(mesh_src)
    
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(mesh_src)
    
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")

    loss = loss_edge * cfg.w_edge + loss_normal * cfg.w_normal + loss_laplacian * cfg.w_laplacian
    
    return loss

class ChamferDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, w1=1.0, w2=1.0, each_batch=False):
        self.check_parameters(points1)
        self.check_parameters(points2)

        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist1 = torch.sqrt(dist1)**2
        dist2 = torch.sqrt(dist2)**2

        dist_min1, indices1 = torch.min(dist1, dim=2)
        dist_min2, indices2 = torch.min(dist2, dim=2)

        loss1 = dist_min1.mean(1)
        loss2 = dist_min2.mean(1)
        
        loss = w1 * loss1 + w2 * loss2

        if not each_batch:
            loss = loss.mean()

        return loss

    @staticmethod
    def check_parameters(points: torch.Tensor):
        assert points.ndimension() == 3  # (B, N, 3)
        assert points.size(-1) == 3