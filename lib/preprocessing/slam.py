import cv2
import numpy as np
import torch
from pathlib import Path
from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg 
from itertools import chain
from tqdm import tqdm

@torch.no_grad()
def run_slam(imagedir, width, height, calib=None, stride=1, skip=0, buffer=2048, gpu_idx=None,
            dpvo_cfg=None, dpvo_ckpt=None):
    if calib is None:
        calib = estimate_intrinsics(width, height)

    cfg.merge_from_file(dpvo_cfg)
    cfg.BUFFER_SIZE = buffer

    slam = None
    # preprocess image chunk according to stride and skip
    
    # image chunk as tensor
    if not isinstance(imagedir, str):
        processed_img, intrinsics = image_chunk(imagedir, calib, stride, skip)
        intrinsics = torch.from_numpy(intrinsics).cuda()

        for t, image in enumerate(processed_img):
            image = torch.from_numpy(image).permute(2,0,1).cuda()

            if slam is None:
                slam = DPVO(cfg, dpvo_ckpt, ht=image.shape[1], wd=image.shape[2], viz=False)

            with Timer("SLAM", enabled=False):
                slam(t, image, intrinsics)

    # image path 
    else:
        intrinsics = np.array(calib[:4])
        intrinsics = torch.from_numpy(intrinsics).cuda()
        img_exts = ["*.png", "*.jpeg", "*.jpg"]
        image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
        
        for t, imfile in tqdm(enumerate(image_list), total=len(image_list)):
            image = cv2.imread(str(imfile))
            image = torch.from_numpy(image).permute(2,0,1).cuda()

            if slam is None:
                slam = DPVO(cfg, dpvo_ckpt, ht=image.shape[1], wd=image.shape[2], viz=False)

            with Timer("SLAM", enabled=False):
                slam(t, image, intrinsics)

        for _ in range(12):
            slam.update()

    result = slam.terminate()
    del slam

    return result

def estimate_intrinsics(width, height):
    # focal_length = (height ** 2 + width ** 2) ** 0.5
    focal_length = max(width, height) * 24./35. # assume equivalent 24mm focal on 35mm sensor
    center_x = width / 2
    center_y = height / 2

    return focal_length, focal_length, center_x, center_y

def image_chunk(imagedir, calib, stride, skip):

    fx, fy, cx, cy = calib[:4]
    intrinsics = np.array([fx, fy, cx, cy])

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy
 
    images = imagedir[skip::stride, ...]
    processed_img = []

    for image in images:
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        processed_img.append(image)

    # concat processed_img into batch
    processed_img = np.stack(processed_img, axis=0)    

    return processed_img, intrinsics