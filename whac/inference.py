import os
import os.path as osp
import argparse
import numpy as np
import math
from tqdm import tqdm
import joblib
import torch
import torchvision.transforms as transforms
import cv2
import datetime
import smplx
from whac.base import Demoer_WHAC
from whac.config import Config as Config_WHAC
from lib.utils.visualization import run_vis_on_demo
from lib.utils.inference_utils import save_mesh_to_obj, process_whac_scale, \
    cam_mesh_to_world_mesh, process_slam_result, smplestx_outputs_to_whac_inputs

# preprocessing
from lib.preprocessing.tracking import YOLOPersonTracker
from lib.preprocessing.slam import run_slam

# smplest-x
from human_models.human_models import SMPLX
from main.base import Tester
from main.config import Config
from utils.data_utils import process_bbox, generate_patch_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str, default='skateboard/city07.mp4')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_mesh', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # prepare output directories
    video = f'./demo/{args.seq_name}'
    video_name = os.path.basename(video).split('.')[0]
    output_folder = os.path.join('./demo/', video_name)
    os.makedirs(output_folder, exist_ok=True)
    # copy input video to output folder
    os.system(f'cp {video} {output_folder}/input.mp4')
    
    slam_res_path = os.path.join(output_folder,'slam_results.pth')
    det_res_path = os.path.join(output_folder,'detection_results.pth')
    
    if args.save_mesh:
        mesh_path = os.path.join(output_folder, 'mesh')
        os.makedirs(mesh_path, exist_ok=True)

    # =============smplestx=============
    smplestx_config_path = './third_party/SMPLest-X/pretrained_models/smplest_x_h/config_base.py'
    smplestx_ckpt_path = './third_party/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar'
    
    # init config
    cfg = Config.load_config(smplestx_config_path)
    exp_name = f'inference_{video_name}_{time_str}'
    new_config = {
        "model": {
            "pretrained_model_path": smplestx_ckpt_path,
            "human_model_path": './third_party/SMPLest-X/human_models/human_model_files'
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join('./outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init smplestx tester
    smplestx = Tester(cfg)
    print(f'Inference [{video}] with [{cfg.model.pretrained_model_path}].')
    smplestx._make_model()
    smplestx.model.eval()

    # =============whac=============
    whac_ckpt_path = './pretrained_models/whac_motion_velocimeter.pth.tar'
    whac_config_path = './configs/config_whac.py'
    cfg_whac = Config_WHAC.load_config(whac_config_path)
    
    new_config = {
        "model": {
            "pretrained_model_path": whac_ckpt_path,
            # "human_model_path": './lib/human_models/human_model_files'
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join('./outputs', exp_name, 'log'),  
            }
    }
    cfg_whac.update_config(new_config)

    whac = Demoer_WHAC(cfg_whac)
    print(f'Inference [{video}] with [{whac_ckpt_path}].')
    whac._make_model()
    whac.model.eval()
    print('Model initialization...Done')

    # =============dpvo=============
    dpvo_cfg = osp.join('./third_party/DPVO', 'config/default.yaml')
    dpvo_ckpt = osp.join('./third_party/DPVO', 'pretrained_models/dpvo.pth')

    # =============load video=============
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    # fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    images = []
    vis_images = None

    # from cap to images
    for i in range(length):
        flag, img = cap.read()
        if not flag: break
        images.append(img)
    images = np.stack(images,axis=0) 
    cap.release()

    # =============preprocessing=============
    if os.path.exists(det_res_path) and os.path.exists(slam_res_path):
        print(f'Load detecion and SLAM results from {output_folder}')
        ### load slam extrinsics and bbox results
        slam_results = joblib.load(os.path.join(output_folder,'slam_results.pth'))
        detection_results = joblib.load(os.path.join(output_folder,'detection_results.pth'))
    
    else:
        print('No detection and SLAM results found. Running Detection and SLAM...')
        # Detection and tracking
        cap = cv2.VideoCapture(video)
        tracker = YOLOPersonTracker(cap, 
                        output_path=osp.join(output_folder,"traking_results.mp4"))
        vis_images, detection_results = tracker.run(save_output=True)
        vis_images = np.stack(vis_images,axis=0) 
        print('Detection and tracking...Done')
        joblib.dump(dict(detection_results), osp.join(output_folder, 'detection_results.pth'))

        # SLAM
        slam_results = run_slam(images, width, height, calib=None, stride=1, skip=0, buffer=2048,
                                dpvo_cfg=dpvo_cfg, dpvo_ckpt=dpvo_ckpt)[0]
        print('SLAM...Done')
        joblib.dump(slam_results, osp.join(output_folder, 'slam_results.pth'))

        print(f'Save processed data at {output_folder}')

    print('Detection and SLAM...Done')

    # only keep the track id with the most detections
    pid = None
    max_length = -1
    for id in detection_results.keys():
        length = len(detection_results[id]['bbox_xyxy'])
        if length > max_length:
            max_length = length
            pid = id
    bbox_xywh = detection_results[pid]['bbox_xywh']
    bbox_frame_id = detection_results[pid]['frame_id']

    # estimate intrinsics
    focal_ = max(width, height) * 24./35. # assume equivalent 24mm focal on 35mm sensor
    focal = np.array([focal_, focal_])
    pcp_pt = np.array([width / 2, height / 2])
    
    ### process slam
    extrinsics = process_slam_result(slam_results)
    
    cropped_img = []
    bboxes = []
    for i in range(len(images)):
        if not i in bbox_frame_id:
            # skip the frame without detection
            continue
        bbox_id = bbox_frame_id.index(i)
        # prepare input image
        original_img = images[i]
        # xywh
        bbox = process_bbox(bbox=bbox_xywh[bbox_id],
                            img_width=width, 
                            img_height=height, 
                            input_img_shape=cfg.model.input_img_shape, 
                            ratio=1.0)
        img, _, _ = generate_patch_image(cvimg=original_img, 
                                            bbox=bbox, 
                                            scale=1.0, 
                                            rot=0.0, 
                                            do_flip=False, 
                                            out_shape=cfg.model.input_img_shape)
        transform = transforms.ToTensor()
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]

        cropped_img.append(img)
        bboxes.append(bbox)

    # =============inference=============
    full_seq_len = len(cropped_img)
    full_loop =int(math.ceil(full_seq_len / cfg_whac.inference.seq_len)) * cfg_whac.inference.seq_len
    cropped_img = torch.stack(cropped_img).squeeze()
    bboxes = torch.tensor(np.stack(bboxes)).cuda()
    extrinsics_rot_mat = torch.tensor(extrinsics['slam_res_rotmat']).cuda()
    
    whac_scale = []
    mesh_seq = []
    for idx in range(0, full_loop, cfg_whac.inference.seq_len): 
        end_idx = idx + cfg_whac.inference.seq_len
        end_idx = min(end_idx, full_seq_len)
        batch_seq_len = end_idx - idx

        inputs = {'img': cropped_img[idx:end_idx],
                'slam_extrinsics': extrinsics_rot_mat[idx:end_idx],}

        targets = {'body_bbox': bboxes[idx:end_idx]}

        meta_info = {'height':torch.tile(torch.tensor(height), (batch_seq_len, )),
                    'width':torch.tile(torch.tensor(width), (batch_seq_len, )),
                    'principal_pt': torch.tile(torch.tensor(pcp_pt), (batch_seq_len, 1)),
                    'focal_length': torch.tile(torch.tensor(focal), (batch_seq_len, 1))}
        
        with torch.no_grad():
            # smplestx
            out_smplestx = smplestx.model(inputs, targets, meta_info, 'test')
            inputs_whac = smplestx_outputs_to_whac_inputs(cfg, out_smplestx, inputs, targets, meta_info)
            
            # whac
            out = whac.model(inputs_whac, targets, meta_info, 'test')

        whac_scale.extend([item for sublist in out['whac_scale'].tolist() for item in sublist])
        mesh_seq.extend(out['smplx_mesh_cam_gt_focal'])

    # use the full video average scale rather than per time step
    mesh_seq = torch.stack(mesh_seq)
    whac_scale = process_whac_scale(whac_scale)
    # transform mesh to world accorcing to sacaled extrinsics
    world_mesh, extrinsics_rot_mat = \
            cam_mesh_to_world_mesh(bbox_frame_id, mesh_seq, extrinsics_rot_mat, whac_scale)
    print("SMPLest-X and WHAC...Done")

    # =============visualization=============
    layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 
                'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 
                'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
    layer= smplx.create(cfg_whac.model.human_model_path, 'smplx', gender='NEUTRAL', 
                use_pca=False, use_face_contour=True, **layer_arg)

    # save mesh
    if args.save_mesh:
        for i in tqdm(range(world_mesh.shape[0])):
            verts = world_mesh[i]
            save_mesh_to_obj(osp.join(mesh_path, f'whac_results_{i:06}.obj'), verts, layer.faces)

    if args.visualize:
        print("Visualizing...")
        
        with torch.no_grad():
            run_vis_on_demo(video, bbox_frame_id, world_mesh, mesh_seq, focal, pcp_pt, 
                            extrinsics_rot_mat, output_folder, layer.faces)


if __name__ == "__main__":
    main()