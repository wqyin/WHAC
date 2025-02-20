import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
from lib.utils.renderer import Renderer, get_global_cameras

def run_vis_on_demo(video, frame_id, world_results, cam_result, focal, pcp_pt, extrinsics, output_pth, face, vis_global=True):
    extrinsics = extrinsics.float()
    world_results = world_results.float()
    cam_result = cam_result.float()

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # convert cap to list of images
    images = []
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag: break
        images.append(img[..., ::-1])
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, 'cuda', face)

    if vis_global:
        verts_glob = world_results.cpu()
        offset = verts_glob[..., 1].min().item()
        # verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(max(sx.item(), sz.item()) * 1.5, 10)
        
        # set default ground
        renderer.set_ground(scale, cx.item(), cz.item(), offset=offset)
        renderer.set_cam_mesh(extrinsics)

        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, 'cuda', 
                                    distance=scale, position=(-scale*0.8, scale*0.8, 0))
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        osp.join(output_pth, 'whac_results.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
    )
    bar = Bar('Rendering results ...', fill='#', max=length)
    
    for i, id in enumerate(frame_id):
        verts = verts_glob[i]
        renderer.create_camera(default_R, default_T)

        img = renderer.render_mesh(focal, pcp_pt, cam_result[i], images[id])
    
        if vis_global:
            # render the global coordinate
            verts = verts.to('cuda').unsqueeze(0)
            faces = renderer.faces.clone().squeeze(0)
            colors = torch.ones((1, 4)).float().to('cuda'); colors[..., :3] *= 0.9
            
            cameras = renderer.create_camera(global_R[i], global_T[i])
            img_glob = renderer.render_with_ground(i, verts, faces, colors, cameras, global_lights)

            try: img = np.concatenate((img, img_glob), axis=1)
            except: img = np.concatenate((img, np.ones_like(img) * 255), axis=1)

        writer.append_data(img)
        bar.next()

    writer.close()