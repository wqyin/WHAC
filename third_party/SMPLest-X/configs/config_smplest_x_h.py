config = {
  "data": {
    "use_cache": True,
    "data_dir": "./data",
    "trainset_humandata": [
          'SynHand'],
    "testset": 'EHF',
    "BEDLAM_train_sample_interval": 5,
    "SynBody_train_sample_interval": 10,
    "EgoBody_Kinect_train_sample_interval": 10,
    "UBody_train_sample_interval": 10,
    "MPI_INF_3DHP_train_sample_interval": 5,
    "InstaVariety_train_sample_interval": 10,
    "RenBody_HiRes_train_sample_interval": 5,
    "ARCTIC_train_sample_interval": 10,
    "RenBody_train_sample_interval": 10,
    "Talkshow_train_sample_interval": 10,
    "bbox_ratio": 1.2,
    "no_aug": False,
    "data_strategy": "balance",
    "total_data_len": 7500000,
  },

  "train": {
    "num_gpus": 1,
    "continue_train": True,
    "start_over": True,
    "end_epoch": 20,
    "train_batch_size": 16,
    "num_thread": 1,
    "lr": 1e-5,
    "min_lr": 1e-6,
    "save_epoch": 1,
    "remove_checkpoint": False,
    "print_iters": 100,
    "smplx_kps_3d_weight": 100.0,
    "smplx_kps_2d_weight": 1.0,
    "smplx_pose_weight": 10.0,
    "smplx_shape_weight": 1.0,
    "smplx_orient_weight": 1.0,
    "hand_root_weight": 1.0,
    "hand_consist_weight": 1.0,
  },

  "inference":{
    "num_gpus": 1,
    "detection":{
      "model_type": "yolo",
      "model_path": "./pretrained_models/yolov8x.pt",
      "conf": 0.5,
      "save": False,
      "verbose": False,
      "iou_thr": 0.5,
    },
  },

  "test": {
    "test_batch_size": 1
  },

  "model": {
    'model_type': 'vit_huge',
    "pretrained_model_path": './outputs/train_annot_xtp20_20241108_203832/model_dump/snapshot_7.pth.tar',
    "human_model_path": './human_models/human_model_files',
    'encoder_pretrained_model_path': './pretrained_models/vitpose_huge.pth',
    'encoder_config': {
        'num_classes': 80,
        'task_tokens_num': 80,
        'img_size': (256, 192),
        'patch_size': 16,
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'ratio': 1,
        'use_checkpoint': False,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'drop_path_rate': 0.55
    },
    'decoder_config': {
      'feat_dim': 1280,
      "dim_out": 512,
      'task_tokens_num': 80,
    },
    'input_img_shape': (512, 384),
    'input_body_shape': (256, 192),
    'output_hm_shape': (16, 16, 12),
    'focal': (5000, 5000),
    'princpt': (192 / 2, 256 / 2),  # virtual principal point position
    'body_3d_size': 2,
    'hand_3d_size': 0.3,
    'face_3d_size': 0.3,
    'camera_3d_size': 2.5,
  },
  
  "log":{
      'exp_name': None,
      'output_dir': None,
      'model_dir': None,
      'log_dir': None,
      'result_dir': None,
  }
}