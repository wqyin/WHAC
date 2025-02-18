import torch
import torch.nn as nn
import numpy as np
import smplx
from lib.utils.transforms import batch_rodrigues, invert_transformation
from lib.utils.inference_utils import global_align_joints

# naive version
class MotionPrior(nn.Module):
    def __init__(self, n_kps, feature_size, hidden_size=512, num_layers=3, output_size=3):
        super(MotionPrior, self).__init__()

        self.pose_embed = nn.Linear(n_kps * 3, feature_size)
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, keypoints3d):
        bs, seqlen = keypoints3d.shape[:2]
        x = self.pose_embed(keypoints3d.reshape(bs, seqlen, -1))
        x, _ = self.gru(x)
        out = self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))
        return out
        
class Model(nn.Module):
    def __init__(self, motion_prior, human_model_path):
        super(Model, self).__init__()
        self.whac = motion_prior

        # human model
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 
                        'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 
                        'create_expression': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(human_model_path, 'smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'male': smplx.create(human_model_path, 'smplx', gender='MALE', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'female': smplx.create(human_model_path, 'smplx', gender='FEMALE', use_pca=False, use_face_contour=True, **self.layer_arg)
                        }
        self.J_regressor = self.layer['neutral'].J_regressor.numpy()
        self.joint_mapper = [1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
    
    def forward(self, inputs, targets, meta_info, mode):
        if mode == "train": 
            raise NotImplementedError("Training is not yet implemented.")
        else:
            cam_mesh = inputs['smplx_mesh_cam_gt_focal'].squeeze()
            ext_rot_mat = inputs['slam_extrinsics'].squeeze()
            smplx_rotation_aa = inputs['smplx_root_pose'].squeeze()
            transl = inputs['cam_trans_gt_focal'].squeeze()
            world2cam_flag = False

        if cam_mesh.shape[1] == 10475:
            cam_kps3d = torch.matmul(torch.Tensor(self.J_regressor).cuda(), 
                                    cam_mesh.to(dtype=torch.float32))[:, [0] + self.joint_mapper, :] # nkps=15
        
        cam_kps3d_homo = torch.cat((cam_kps3d, torch.ones((cam_kps3d.shape[0], cam_kps3d.shape[1], 1)).cuda()), axis=2)
        
        # cam extrinsics
        if world2cam_flag:
            raise NotImplementedError("Training is not yet implemented.")
        else:
            # cam2world for slam prediction
            start_frame = invert_transformation(ext_rot_mat[0]).double().cuda()
            ext_rot_mat_offset = torch.stack([torch.matmul(start_frame, matrix) for matrix in ext_rot_mat]).cuda()

        # smplx in cam space
        smplx_rotation_mat = batch_rodrigues(smplx_rotation_aa)
        smplx_transl = torch.cat((transl, torch.ones((transl.shape[0], 1)).cuda()), axis=1)
        smplx_transl_homo = torch.zeros_like(ext_rot_mat)
        smplx_transl_homo[:, :3, :3] = smplx_rotation_mat
        smplx_transl_homo[:, :, 3] = smplx_transl

        # in world space
        world_kps3d_homo =torch.einsum('bij,bkj->bki',ext_rot_mat_offset.float(), cam_kps3d_homo)

        # offset and make first frame canonical 
        cano_world_kps3d_homo = torch.einsum('bij,bkj->bki', invert_transformation(smplx_transl_homo[0])[None, ...].float().cuda(), world_kps3d_homo)

        # move all frames to frame 1 canonical space
        cano_world_kps3d_homo_offset = cano_world_kps3d_homo - cano_world_kps3d_homo[:, 0, :][:, None, :] # all offset to pelvis at origin

        out = self.whac(cano_world_kps3d_homo_offset[None, :, :, :3]) #, est_t_per_frame_norm[None, ...])
        
        # full traj from origin
        scaled_slam_traj = torch.cumsum(out, dim=1) # [bs, seq, t] per frame

        # human_traj_world = cano_world_kps3d_homo_offset[None, :, :, :3] + scaled_traj[:, :, None, :]
        human_slam_traj_world = cano_world_kps3d_homo_offset[None, :, :, :3] + scaled_slam_traj[:, :, None, :]


        if mode == "train":
            raise NotImplementedError("Training is not yet implemented.")

        else:
            
            T_h2w = torch.stack([torch.matmul(mat_c2w, mat_h2cam) for (mat_c2w, mat_h2cam) in zip(ext_rot_mat_offset.float(), smplx_transl_homo.float())])
            T_h2cano = torch.stack([torch.matmul(invert_transformation(smplx_transl_homo[0]).float().cuda(), mat_h2w) for mat_h2w in T_h2w])
            t_cam2h = torch.stack([invert_transformation(mat.float().cuda())[:3, 3] for mat in smplx_transl_homo])
            whac_t = torch.einsum('bij,bj->bi',T_h2cano[:, :3, :3], t_cam2h.cuda()) + scaled_slam_traj.squeeze() + cano_world_kps3d_homo[0, 0, :3][ None, :]

            out = {}
            whac_scale = []
            n = 5
            for i in range(whac_t.shape[0]-n+1):
                
                scale = global_align_joints(whac_t[i:i+n,None,:3].float().detach().cpu(), ext_rot_mat_offset[i:i+n, None,:3, 3].float().detach().cpu())[1]
                scale = np.nan_to_num(scale.numpy(), nan=0)
                whac_scale.append(scale[0])
                
            for j in range(n-1):
                whac_scale.append(np.nan)

            out['whac_scale'] = torch.tensor(whac_scale).view(-1, 1) #whac_scale.repeat(inputs['slam_extrinsics'].shape[0], 1)
            whac_scale = np.nanmedian(np.array(whac_scale))
            whac_scale = torch.tensor(whac_scale)           

            scaled_slam_extrinsics = ext_rot_mat_offset.detach()
            scaled_slam_extrinsics[:, :3, 3] *= whac_scale.cuda()
            # convert to original cordinate (cam0 of whole sequence)
            scaled_slam_extrinsics = torch.matmul(ext_rot_mat[0][None, ...], scaled_slam_extrinsics)
            
            # convert cano space human traj to word space
            human_slam_traj_cano2local_world = torch.einsum('ij,bkj->bki', smplx_transl_homo[0].float()[:3, :3], human_slam_traj_world.squeeze()) + smplx_transl_homo[0].float()[:3, 3]
            human_slam_traj_cano2global_world = torch.einsum('bij,bkj->bki', ext_rot_mat[0].float()[None, :3, :3], human_slam_traj_cano2local_world.float()) + ext_rot_mat[0][None, :3, 3]
             
            out['world_mp_human_traj'] = human_slam_traj_cano2global_world# homo # whac output with mp, wo slam, in global world space
            out['world_cano_scaled_slam_human_traj'] = human_slam_traj_world.squeeze() # homo # output with mp, wo slam
            out['scaled_slam_extrinsics'] = scaled_slam_extrinsics # homo # scale with mp+slam
            out['whac_mp_cam_traj'] = whac_t[:, :3].float() # cam, mp only

            out['smplx_mesh_cam'] = inputs['smplx_mesh_cam'] # do not use intrinsics
            out['smplx_mesh_cam_gt_focal'] = inputs['smplx_mesh_cam_gt_focal'] # use intrinsics
            out['slam_extrinsics'] = inputs['slam_extrinsics']

            return out

def get_model_whac(mode, cfg):

    motion_prior = MotionPrior(n_kps=cfg.motion_prior.n_kps, 
                            feature_size=cfg.motion_prior.feature_size, 
                            hidden_size=cfg.motion_prior.hidden_size, 
                            num_layers=cfg.motion_prior.num_layers, 
                            output_size=cfg.motion_prior.output_size)
    
    model = Model(motion_prior, cfg.model.human_model_path)

    return model