env

export PATH="/data/anaconda3/bin:$PATH"
export TORCH_HOME='/data/.torch/'

cd /data/task/Detection/dev/FaceBoxes.PyTorch-master

CUDA_VISIBLE_DEVICES=0 python test_s3fd_wider.py --trained_model weights/S3FD_test_6/Final_S3FD.pth