config = {
  "model": {
    "pretrained_model_path": './pretrained_models/whac_motion_velocimeter.pth.tar',
    "human_model_path": './third_party/SMPLest-X/human_models/human_model_files',
  },
  
  "motion_prior":{
    "n_kps": 15, 
    "feature_size":256, 
    "hidden_size":64, 
    "num_layers":3, 
    "output_size":3
  },
  "inference":{
    "seq_len": 64,
  },

  "log":{
      'exp_name': None,
      'output_dir': None,
      'model_dir': None,
      'log_dir': None,
      'result_dir': None,
  }
}