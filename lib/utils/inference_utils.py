import torch
import numpy as np
from lib.utils.transforms import quat_to_rotmat

def align_pcl(Y, X, weight=None, fixed_scale=False):
    """align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
                dim=-1, keepdim=True
            )
            / var[..., 0]
        )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t

def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(
        gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3)
    )
    pred_first = (
        s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    )
    return pred_first

# The functions below are borrowed from SLAHMR official implementation.
# Reference: https://github.com/vye16/slahmr/blob/main/slahmr/eval/tools.py
def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(
        gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3)
    )
    pred_glob = (
        s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    )
    return pred_glob, s_glob

def save_mesh_to_obj(filename, verts, faces):
   with open(filename, 'w') as obj_file:
       # Write vertices
       for v in verts:
           obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
       # Write faces
       for face in faces:
           obj_file.write(f"f {' '.join(map(str, face+1))}\n")

def process_whac_scale(whac_scale):
    whac_scale = np.array(whac_scale)
    nan_mask = ~np.isnan(whac_scale)
    whac_scale = np.mean(whac_scale[nan_mask])

    print(f"Scale: {whac_scale}")

    return whac_scale

def cam_mesh_to_world_mesh(bbox_frame_id, mesh_seq, extrinsics_rot_mat, whac_scale):
    extrinsics_rot_mat[: , :3, 3] *= whac_scale
    extrinsics_rot_mat = extrinsics_rot_mat.float()
    R = torch.tensor([[-1, 0, 0],[0, -1, 0], [0, 0, 1]]).to('cuda').float()
    extrinsics_rot_mat[:, :3] = torch.einsum('ij,bjk->bik', R, extrinsics_rot_mat[:, :3])
    
    world_mesh = torch.einsum('bij,bkj->bki', extrinsics_rot_mat[bbox_frame_id, :3, :3], mesh_seq) + \
                        extrinsics_rot_mat[bbox_frame_id, :3, 3][:, None, :]

    return world_mesh, extrinsics_rot_mat

def process_slam_result(result):
    video_results = {}
    
    translation = result[:, :3]
    quaternion_xyzw = result[:, 3:]
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    result[:, 3:] = quaternion_wxyz

    rotation = quat_to_rotmat(torch.Tensor(quaternion_wxyz)) # cam2world
    rotation = rotation.numpy()

    RT = np.concatenate([rotation, translation.reshape(-1,3,1)], axis=2) # world2cam
    padding = np.zeros((RT.shape[0], 1, 4))
    padding[:, :, -1] = 1
    RT_homo = np.concatenate((RT, padding), axis=1)

    video_results['slam_res_quat'] = result[0]
    video_results['slam_res_rotmat'] = RT_homo

    return video_results

def smplestx_outputs_to_whac_inputs(cfg, out_smplestx, inputs, targets, meta_info):

    # process mesh in cam space 
    focal_x = cfg.model.focal[0] / cfg.model.input_body_shape[1] * targets['body_bbox'][:,2]
    focal_y = cfg.model.focal[1] / cfg.model.input_body_shape[0] * targets['body_bbox'][:,3]
    full_img_focal = torch.cat((focal_x.view(-1,1), focal_y.view(-1,1)), dim=1)
    
    cam_trans_gt_focal = out_smplestx['cam_trans'].clone()
    pred_depth = cam_trans_gt_focal[:, 2].clone()
    cam_trans_gt_focal[:, 2] *= meta_info['focal_length'][:, 0].cuda() / full_img_focal[:, 0]

    shift_x = (targets['body_bbox'][:, 0] + targets['body_bbox'][:, 2] / 2. - meta_info['width'].cuda() / 2. ) * \
                pred_depth  / full_img_focal[:, 0]
    shift_y = (targets['body_bbox'][:, 1] + targets['body_bbox'][:, 3] / 2. - meta_info['height'].cuda() / 2. ) * \
                pred_depth / full_img_focal[:, 1]
    cam_trans_gt_focal[:, 0] += shift_x
    cam_trans_gt_focal[:, 1] += shift_y

    # smooth cam_trans_gt_focal
    cam_trans_gt_focal = gaussian_smooth(cam_trans_gt_focal, sigma=2, kernel_size=9)

    # shift the smplx mesh to the new depth
    mesh_cam_gt_focal = out_smplestx['smplx_mesh_cam'] \
            - out_smplestx['cam_trans'].unsqueeze(1) + cam_trans_gt_focal.unsqueeze(1)

    out_smplestx['smplx_mesh_cam_gt_focal'] = mesh_cam_gt_focal
    out_smplestx['slam_extrinsics'] = inputs['slam_extrinsics']
    out_smplestx['cam_trans_gt_focal'] = cam_trans_gt_focal

    return out_smplestx


def gaussian_smooth(tensor, sigma=2, kernel_size=9):
    # Create Gaussian kernel
    x = torch.arange(kernel_size, device=tensor.device) - kernel_size // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize

    kernel = kernel.repeat(3, 1, 1)  # Shape: [3, 1, kernel_size]

    # Permute tensor to (batch=1, channels=3, time) format
    tensor = tensor.T.unsqueeze(0)  # Shape: [1, 3, t]

    # Compute padding size
    pad_size = (kernel_size - 1) // 2
    tensor_padded = torch.nn.functional.pad(tensor, (pad_size, pad_size), mode="replicate")

    # Apply 1D convolution with groups=3
    smoothed = torch.nn.functional.conv1d(tensor_padded, kernel, groups=3)

    return smoothed.squeeze(0).T  # Convert back to [t, 3]

