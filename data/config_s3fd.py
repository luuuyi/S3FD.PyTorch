# config_s3fd.py

cfg = {
    'name': 'S3FD',
    'feature_maps': [[160, 160], [80, 80], [40, 40], [20, 20], [10, 10], [5, 5]],
    'min_dim': 640,
    'steps': [4, 8, 16, 32, 64, 128],
    'min_sizes': [[16], [32], [64], [128], [256], [512]],
    'aspect_ratios': [[1], [1], [1], [1], [1], [1]],
    'variance': [0.1, 0.2],
    'clip': False,
    'conf_weight': 4.0,
    'rgb_mean': [104, 117, 123],
    'max_expand_ratio': 4,
}
